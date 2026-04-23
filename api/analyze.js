// ============================================================================
// AI 성취수준 분석 API (Claude Opus 4.7 + Vision/Document)
// ----------------------------------------------------------------------------
// 환경 변수 (Vercel 대시보드 > Settings > Environment Variables):
//   ANTHROPIC_API_KEY  (필수)  Anthropic API 키 (Claude)
//   ADMIN_CODE         (선택)  설정 시 X-Admin-Code 헤더 필수
//
// 핵심 최적화:
//   · 출제원안지 PDF를 document 블록으로 첨부 → cache_control로 캐싱
//     (같은 PDF로 학생 30명 분석 시 PDF 토큰은 첫 요청만 풀가격)
//   · adaptive thinking + effort:"high" (교육 평가는 정확도 우선)
//   · output_config.format: json_schema → 구조화 출력 보장
//   · 스트리밍으로 SDK HTTP 타임아웃 회피 (긴 응답 대비)
// ============================================================================

import Anthropic from '@anthropic-ai/sdk';

const SYSTEM_PROMPT = `너는 고등학교 과학 평가 분석 전문가다.
학생 한 명의 시험 응답을 분석하여 내용 영역별 성취수준을 객관적으로 진술한다.

[입력으로 받는 것]
1) 출제원안지 PDF (첨부) — 실제 문항 내용과 난이도 확인용
2) 문항정보표 (표 형식 텍스트) — 문항번호 · 내용영역 · 성취기준코드 · 난이도 · 배점 · 정답
3) 학생 응답 (익명 ID) — 각 문항에 대한 학생의 답과 정오

[엄격 규칙]
1. 성취기준 코드(예: [10통과1-01-02])를 반드시 인용한다.
2. 학생을 격려·비난하는 표현을 쓰지 않는다. 객관적·중립적 서술만.
3. "~할 수 있다", "~에 대한 이해가 부족하다", "~문항에서 정답을 도출하지 못했다" 등 학습과학 용어를 사용한다.
4. 각 내용 영역 진술문은 3~5문장이고, 구체적 근거(어떤 성취기준 문항에서 맞고/틀렸는지, 출제원안지에서 확인한 개념적 특성)를 포함한다.
5. 성취수준은 배점 대비 득점률로 판정:
   - 상: 80% 이상
   - 중: 50~79%
   - 하: 50% 미만
6. 서답형 문항은 정답이 "별지 참조"/수작업 채점이므로, 배점 정보만 참고하고 구체적 옳고그름은 판단하지 않는다.
7. 출제원안지 PDF의 실제 문항 내용을 참고하여 진술문에 구체적 개념 언급 (예: "규산염 사면체 구조 파악", "빅뱅 우주론에 따른 원소 합성").
8. 응답은 반드시 JSON 스키마를 정확히 따른다 (추가 필드 금지, 누락 금지).`;

const JSON_SCHEMA = {
  type: 'object',
  properties: {
    areas: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          areaName: { type: 'string', description: '내용영역명' },
          totalPoints: { type: 'number', description: '해당 영역 총 배점' },
          scoredPoints: { type: 'number', description: '학생 득점' },
          attainmentPercent: { type: 'number', description: '득점률 0-100' },
          masteryLevel: { type: 'string', enum: ['상', '중', '하'] },
          statement: { type: 'string', description: '3~5문장 객관적 성취수준 진술문' },
          standardsEvaluated: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                code: { type: 'string', description: '성취기준 코드 (대괄호 제외)' },
                hits: { type: 'string', description: '정답 수/전체 수 예: "2/3"' },
                note: { type: 'string', description: '짧은 평가 한 줄' }
              },
              required: ['code', 'hits', 'note'],
              additionalProperties: false
            }
          }
        },
        required: [
          'areaName', 'totalPoints', 'scoredPoints',
          'attainmentPercent', 'masteryLevel', 'statement',
          'standardsEvaluated'
        ],
        additionalProperties: false
      }
    },
    overallNote: {
      type: 'string',
      description: '전 영역 종합 1~2문장 (객관적)'
    }
  },
  required: ['areas', 'overallNote'],
  additionalProperties: false
};

function buildUserPrompt(itemInfo, student) {
  const safeId = String(student.id || student.anonId || 'student').slice(0, 30);

  const lines = [];
  lines.push(`[분석 대상] 학생 ID: ${safeId}`);
  lines.push(`[학급 번호] ${student.classNum || '-'}`);
  lines.push('');
  lines.push('[문항정보표 및 학생 응답]');
  lines.push('문항번호 | 내용영역 | 성취기준코드 | 난이도 | 배점 | 정답 | 학생답 | 정오 | 서답형여부');

  for (const item of itemInfo) {
    const ans = student.answers.find(a => a.q === item.q);
    const studentAns = ans?.response ?? '-';
    const correct = ans ? (ans.correct ? 'O' : 'X') : '-';
    const isEssay = String(item.answer || '').includes('별지') || item.q > 26;
    lines.push(
      `${item.q} | ${item.area} | ${item.standardCode || '-'} | ${item.level} | ${item.point} | ${item.answer} | ${studentAns} | ${correct} | ${isEssay ? 'Y' : 'N'}`
    );
  }

  lines.push('');
  lines.push('[요구사항]');
  lines.push('첨부된 출제원안지 PDF의 실제 문항 내용을 참고하여, 위 학생의 영역별 성취수준을 JSON 스키마에 맞춰 분석하라.');
  lines.push('성취기준 코드별로 어떤 개념이 강점/약점인지 구체적으로 서술하되, 학생을 평가(격려/비난)하지 말고 객관 사실만 서술한다.');

  return lines.join('\n');
}

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', req.headers.origin || '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-Admin-Code');

  if (req.method === 'OPTIONS') { res.status(204).end(); return; }
  if (req.method !== 'POST') { res.status(405).json({ error: 'Method not allowed' }); return; }

  // 선택적 관리자 코드 인증 (ADMIN_CODE 환경변수가 있을 때만)
  const expectedAdmin = process.env.ADMIN_CODE;
  if (expectedAdmin) {
    const provided = req.headers['x-admin-code'] || '';
    if (provided !== expectedAdmin) {
      res.status(401).json({ error: '관리자 코드가 올바르지 않습니다.' });
      return;
    }
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    res.status(500).json({ error: '서버에 ANTHROPIC_API_KEY가 설정되지 않았습니다.' });
    return;
  }

  const { testPaperPdf, itemInfo, student } = req.body || {};

  if (!itemInfo || !Array.isArray(itemInfo) || itemInfo.length === 0) {
    res.status(400).json({ error: '문항정보표(itemInfo)가 필요합니다.' });
    return;
  }
  if (!student || !student.answers || !Array.isArray(student.answers)) {
    res.status(400).json({ error: '학생 응답 데이터(student.answers)가 필요합니다.' });
    return;
  }
  if (!testPaperPdf || typeof testPaperPdf !== 'string' || testPaperPdf.length < 100) {
    res.status(400).json({ error: '출제원안지 PDF(testPaperPdf)가 필요합니다.' });
    return;
  }

  const client = new Anthropic({ apiKey });
  const userText = buildUserPrompt(itemInfo, student);

  try {
    // 스트리밍으로 호출하여 SDK HTTP 타임아웃 회피 (adaptive thinking + PDF는 30s+ 가능)
    const stream = client.messages.stream({
      model: 'claude-opus-4-7',
      max_tokens: 16000,
      thinking: { type: 'adaptive' },
      output_config: {
        effort: 'high',
        format: { type: 'json_schema', schema: JSON_SCHEMA }
      },
      system: SYSTEM_PROMPT,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'document',
              source: {
                type: 'base64',
                media_type: 'application/pdf',
                data: testPaperPdf
              },
              // PDF 블록에 캐시 breakpoint → 같은 시험 여러 학생 분석 시
              // 시스템 프롬프트 + PDF가 캐시되어 이후 요청은 ~0.1배 비용
              cache_control: { type: 'ephemeral' }
            },
            { type: 'text', text: userText }
          ]
        }
      ]
    });

    const message = await stream.finalMessage();

    // JSON 추출 (output_config.format으로 첫 text 블록이 유효한 JSON임이 보장됨)
    const textBlock = message.content.find(b => b.type === 'text');
    if (!textBlock) {
      res.status(500).json({ error: 'AI 응답에서 텍스트 블록을 찾지 못했습니다.' });
      return;
    }

    let parsed;
    try {
      parsed = JSON.parse(textBlock.text);
    } catch (e) {
      res.status(500).json({
        error: 'AI 응답을 JSON으로 파싱할 수 없습니다.',
        raw: textBlock.text.slice(0, 500)
      });
      return;
    }

    // (선택적) 캐시 히트 로깅 — Vercel 로그에서 확인 가능
    const usage = message.usage || {};
    console.log('[analyze] tokens:',
      'in=' + (usage.input_tokens || 0),
      'out=' + (usage.output_tokens || 0),
      'cache_read=' + (usage.cache_read_input_tokens || 0),
      'cache_write=' + (usage.cache_creation_input_tokens || 0)
    );

    res.status(200).json(parsed);

  } catch (error) {
    console.error('Analyze error:', error);

    // SDK의 타입드 예외 사용
    if (error instanceof Anthropic.RateLimitError) {
      res.status(429).json({ error: 'API 호출 한도 초과. 잠시 후 다시 시도해주세요.' });
    } else if (error instanceof Anthropic.AuthenticationError) {
      res.status(500).json({ error: 'ANTHROPIC_API_KEY가 올바르지 않습니다.' });
    } else if (error instanceof Anthropic.APIError) {
      res.status(error.status || 500).json({
        error: `Claude API 오류 (${error.status}): ${error.message || ''}`
      });
    } else {
      res.status(500).json({ error: error.message || '서버 내부 오류' });
    }
  }
}

export const config = {
  api: {
    bodyParser: { sizeLimit: '12mb' },
    responseLimit: false
  }
};
