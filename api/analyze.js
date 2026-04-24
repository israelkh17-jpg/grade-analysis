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
학생 한 명의 내용 영역별 성취수준을 **간결하게** 진술한다.

[입력]
1) 출제원안지 PDF — 문항 내용 확인용
2) 문항정보표 — 문항번호 · 내용영역 · 성취기준 코드/전문 · 난이도 · 배점 · 정답
3) 학생 응답 — 각 문항 정오

[진술 스타일 — 평가기준 A~E 스타일, 간결체]
성취기준 전문의 핵심 개념·행위 동사를 재사용하여 "~을 (인식/구분/설명/추론)할 수 있다" 형태로 기술.
추상 표현("상/중/하 수준", "이해가 안정적", "보완 필요" 등) 단독 사용 금지.
등급 라벨(A/B/C/D/E, 상/중/하)은 statement 본문에 노출 금지 (masteryLevel 필드에만 기록).

[영역별 statement — 2~3문장, 각 250~350자 범위]
① 역량 진술: 성취기준의 핵심 개념·행위 동사를 풀어 "~할 수 있다"로 인정.
   (예: "세포 내 정보가 DNA에 유전자로 저장되어 있음을 인식하고, 전사·번역을 통해 단백질이 생성되는 흐름을 설명할 수 있다.")
② (오답 있을 때) 한계: 어느 문항에서 어떤 개념의 연결이 미완결인지 "~하는 데에는 이르지 못하였다" 형태로.
   (예: "다만 7번 문항에서 조절 유전자 사례의 연결은 완결적으로 기술하는 데에는 이르지 못하였다.")

[총 분량 가이드] 모든 영역 statement를 합쳐 **1000~1500자** 범위에 맞춘다. 과도한 부연 금지.

[masteryLevel 판정] 배점 대비 득점률 — 상 80%+ / 중 50~79% / 하 50% 미만.

[standardsEvaluated[].note] 한 줄, 25자 내외. 구체 역량 키워드로(예: "DNA→단백질 흐름 주요 단계 기술").

[overallNote] 1문장, 80자 이내.

반드시 JSON 스키마를 정확히 따른다(추가 필드 금지, 누락 금지).`;

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
          statement: { type: 'string', description: '2~3문장 성취수준 진술문(영역당 250~350자). 평가기준 A~E 스타일의 구체 역량 기술. "상/중/하 수준" 같은 라벨 본문 노출 금지.' },
          standardsEvaluated: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                code: { type: 'string', description: '성취기준 코드 (대괄호 제외)' },
                hits: { type: 'string', description: '정답 수/전체 수 예: "2/3"' },
                note: { type: 'string', description: '짧은 평가 한 줄. "상 수준" 같은 추상 표현 금지. 구체 역량(예: "DNA→RNA→단백질 흐름의 주요 단계 서술 가능")으로 작성.' }
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

  // 성취기준 코드별 전문(standardText) — 평가기준 스타일 진술 구성의 근거 자료
  const stdMap = new Map();
  for (const item of itemInfo) {
    const code = item.standardCode || '';
    if (!code) continue;
    if (!stdMap.has(code) && item.standardText) stdMap.set(code, item.standardText);
  }
  if (stdMap.size > 0) {
    lines.push('');
    lines.push('[성취기준 전문 — 진술문 구성 시 핵심 개념·과정·관계의 근거]');
    for (const [code, text] of stdMap.entries()) {
      lines.push(`${code}: ${text}`);
    }
  }

  lines.push('');
  lines.push('[요구사항] 위 자료를 근거로 영역별 성취수준을 JSON 스키마에 맞춰 분석하라.');
  lines.push('- statement: 영역당 2~3문장, 각 250~350자. 성취기준의 행위 동사("설명할 수 있다" 등)를 재사용한 평가기준 A~E 스타일 구체 역량 문장.');
  lines.push('- 오답은 "이르지 못하였다" 류 행동 기반 표현. "상/중/하/A~E" 라벨은 본문에 노출 금지(masteryLevel 필드에만 기록).');
  lines.push('- 모든 영역 statement 합계는 1000~1500자 범위에 맞출 것. 과도한 부연 금지.');

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

  // ── SSE 모드로 응답 ─────────────────────────────────────────────
  // 긴 생성(>60s)에도 중간 프록시가 끊지 않도록, Anthropic 스트림 이벤트마다
  // progress 이벤트를 내려보내 커넥션을 유지하고, 마지막에 result 이벤트로 JSON 전송.
  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no'); // nginx/edge 버퍼링 차단
  if (typeof res.flushHeaders === 'function') res.flushHeaders();

  const send = (event, data) => {
    try {
      res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
    } catch (_) { /* 연결 끊김 무시 */ }
  };

  // 15s 하트비트: 이벤트 공백이 길 때도 커넥션 유지
  const heartbeat = setInterval(() => {
    try { res.write(': ping\n\n'); } catch (_) {}
  }, 15000);

  send('progress', { stage: 'start', message: 'AI 분석 시작' });

  const client = new Anthropic({ apiKey });
  const userText = buildUserPrompt(itemInfo, student);

  try {
    const stream = client.messages.stream({
      model: 'claude-sonnet-4-6',
      max_tokens: 3000,
      thinking: { type: 'enabled', budget_tokens: 1024 },
      output_config: {
        effort: 'medium',
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
              // 같은 시험 여러 학생 분석 시 시스템+PDF가 캐싱되어 이후 요청은 저비용
              cache_control: { type: 'ephemeral' }
            },
            { type: 'text', text: userText }
          ]
        }
      ]
    });

    // 스트림 이벤트 순회 — 진행 상황 주기 전송
    let textChars = 0;
    let thinkingChars = 0;
    let lastSentAt = Date.now();
    const maybeSend = (stage) => {
      const now = Date.now();
      if (now - lastSentAt < 800) return;
      lastSentAt = now;
      send('progress', { stage, textChars, thinkingChars });
    };

    for await (const event of stream) {
      if (event.type === 'content_block_delta') {
        const d = event.delta || {};
        if (d.type === 'thinking_delta') {
          thinkingChars += (d.thinking || '').length;
          maybeSend('thinking');
        } else if (d.type === 'text_delta') {
          textChars += (d.text || '').length;
          maybeSend('writing');
        }
      } else if (event.type === 'content_block_start') {
        const bt = event.content_block?.type;
        if (bt === 'thinking') send('progress', { stage: 'thinking', textChars, thinkingChars });
        else if (bt === 'text') send('progress', { stage: 'writing', textChars, thinkingChars });
      }
    }

    const message = await stream.finalMessage();

    const textBlock = message.content.find(b => b.type === 'text');
    if (!textBlock) {
      send('error', { error: 'AI 응답에서 텍스트 블록을 찾지 못했습니다.' });
      return;
    }

    let parsed;
    try {
      parsed = JSON.parse(textBlock.text);
    } catch (e) {
      send('error', { error: 'AI 응답을 JSON으로 파싱할 수 없습니다.', raw: textBlock.text.slice(0, 500) });
      return;
    }

    const usage = message.usage || {};
    console.log('[analyze] tokens:',
      'in=' + (usage.input_tokens || 0),
      'out=' + (usage.output_tokens || 0),
      'cache_read=' + (usage.cache_read_input_tokens || 0),
      'cache_write=' + (usage.cache_creation_input_tokens || 0)
    );

    send('result', parsed);
  } catch (error) {
    console.error('Analyze error:', error);
    let msg = error.message || '서버 내부 오류';
    if (error instanceof Anthropic.RateLimitError) msg = 'API 호출 한도 초과. 잠시 후 다시 시도해주세요.';
    else if (error instanceof Anthropic.AuthenticationError) msg = 'ANTHROPIC_API_KEY가 올바르지 않습니다.';
    else if (error instanceof Anthropic.APIError) msg = `Claude API 오류 (${error.status}): ${error.message || ''}`;
    send('error', { error: msg });
  } finally {
    clearInterval(heartbeat);
    try { res.end(); } catch (_) {}
  }
}

export const config = {
  api: {
    bodyParser: { sizeLimit: '12mb' },
    responseLimit: false
  }
};
