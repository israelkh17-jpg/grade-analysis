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
2) 문항정보표 (표 형식 텍스트) — 문항번호 · 내용영역 · 성취기준코드 · 성취기준 전문 · 난이도 · 배점 · 정답
3) 학생 응답 (익명 ID) — 각 문항에 대한 학생의 답과 정오

[진술 스타일 — 매우 중요]
진술문은 반드시 **2015 개정 과학과 "평가기준 A~E" 스타일**의 **구체적 역량 기술**로 작성한다.
다음과 같은 추상 표현을 **단독 사용**하거나 **문장의 주어구로** 사용하는 것을 금지한다.
  · "상/중/하 수준이다" · "전반적으로 안정되었다" · "이해가 우수하다"
  · "부분적 오개념이 관찰된다" · "개념 학습이 필요하다" · "보완이 필요하다"
  · "전반적 성취도가 높다/낮다" 등 요약 평어
대신 성취기준 전문(standardText)에 담긴 **행위 동사와 핵심 개념**을 재사용하여,
학생이 "무엇을 인식하고 / 무엇을 구분하고 / 무엇을 연결하여 / 무엇을 설명할 수 있는지"를
능동 동사("인식할 수 있다", "구분할 수 있다", "설명할 수 있다", "추론할 수 있다", "해석할 수 있다" 등)로 풀어 기술한다.

부적합 예시:
  "생명 시스템 영역에서 정답을 도출하여 이해가 전반적으로 안정되었다. 상 수준이다."
적합 예시 (평가기준 A~E 스타일):
  "생명 시스템의 유지에 필요한 세포 내 정보가 DNA에 유전자로 저장되어 있음을 인식하고,
   전사와 번역을 통해 유전자의 염기 서열 정보가 아미노산 서열로 변환되어
   단백질이 생성되는 세포 내 정보 흐름의 체계적 구성을 설명할 수 있다."
(핵심 개념 → 단계 → 연결 관계 순으로 풀어 쓰고, "할 수 있다"로 수준을 함의)

[진술문 구조 — 영역당 3~5문장, 아래 순서를 기본으로 따른다]
① **역량 진술 문장**: 학생이 정답을 도출한 성취기준에 대해, 성취기준 전문의 동사·개념을 재사용한
   평가기준 A~E 스타일의 **구체 역량 문장**. (예: "○○을 인식하고 △△를 설명할 수 있다.")
② **근거 문장**: 어느 문항(번호)에서 어떤 개념을 올바르게 파악했는지 — 출제원안지에서 확인되는
   구체 선지·개념을 인용. (예: "3번 문항에서 명반응과 암반응을 구분하여 올바르게 응답하였다.")
③ **한계 문장**: 오답/미도출 문항이 드러내는 **미연결 개념**을 사실 기반으로 기술. 단정적 오개념 판정 금지.
   (예: "다만 7번 문항에서 광합성과 세포호흡의 에너지 흐름을 정교하게 연결하는 데에는 이르지 못하였다.")
④ (선택) **난이도 참조 문장**: 쉬움/보통/어려움별 정답 양상이 의미하는 바를 짧게.
   (예: "어려움 난이도 문항에서도 정답을 도출하여 단계 간 인과를 연결하는 역량이 확인된다.")
⑤ **마무리 문장**: ①의 역량 진술을 재확인하거나, 복수 성취기준이 있을 때 두 번째 성취기준 역량 진술로 대체.
   라벨("상 수준", "A 등급" 등) 표현 금지.

[평가기준 A~E 해석 가이드 — 정답 양상 × 난이도 → 역량 수준]
성취기준 전문에 등장하는 **핵심 개념/단계/관계**를 {c1, c2, c3, ...}라 할 때:
  · A (최상위): 모든 핵심 개념을 인식하고 단계·인과를 **체계적으로 연결하여** 설명할 수 있다.
    → 적용: 해당 성취기준 전 문항 정답 + **어려움 난이도 정답 포함**.
  · B (상위):   핵심 개념을 인식하고 **전반적 과정**을 설명할 수 있다(일부 하위 단계 생략 가능).
    → 적용: 다수 정답이나 어려움 난이도 일부 오답, 또는 연결형 문항 일부 오답.
  · C (기본):   핵심 개념을 인식하고 **주요 단계를 말할 수 있다**(과정의 핵심만).
    → 적용: 쉬움·보통 난이도 정답 위주, 어려움 난이도에서 오답.
  · D (부분):   핵심 개념은 인식하나 **연결·상세 단계 설명에 한계**가 있다.
    → 적용: 보통 난이도 부분 정답, 다수 오답.
  · E (기초):   **개별 개념의 존재**는 말할 수 있으나 과정·관계 설명에는 이르지 못한다.
    → 적용: 쉬움 난이도에서만 정답, 보통·어려움에서 오답.
등급 라벨(A/B/C/D/E, 상/중/하)은 statement 본문에 **노출하지 않는다**. 반드시 구체 역량 문장으로 풀어 쓴다.

[섬세한 서술 규칙 — 반드시 지킬 것]
1. 성취기준 코드(예: [10통과1-01-02])를 반드시 인용한다.
2. **정답 도출 문항**: 성취기준 전문의 **행위 동사**("설명할 수 있다", "구분할 수 있다", "추론할 수 있다", "해석할 수 있다" 등)를
   그대로 재사용하여 역량을 인정하는 문장을 쓴다. 성취기준 전문의 표현을 최대한 살린다.
3. **오답 문항**: "틀렸다"·"부족하다"·"오개념이 있다"라는 단정 대신,
   "정교하게 연결하는 데 이르지 못하였다", "상세한 ○○의 단계를 구분하는 데 한계를 보였다",
   "○○과 △△의 인과 관계를 완결적으로 기술하는 데에는 이르지 못하였다" 같은
   **사실 중심·행동 기반 표현**을 사용한다.
4. **복수 성취기준 처리**: 한 내용 영역에 여러 성취기준이 포함된 경우,
   **성취기준 코드별로 구분하여** 각각 최소 한 문장 이상 역량 또는 한계를 명시한다.
   여러 성취기준을 묶어 뭉뚱그린 일반 서술은 금지한다.
5. **선택지 인용**: 객관식 문항은 출제원안지 PDF의 **선지 내용**까지 확인하여,
   학생이 고른 선지 또는 정답 선지가 함의하는 **구체 개념**을 진술에 반영한다.
   (예: "규산염 사면체의 공유 결합과 사면체 배열을 구분하는 선지에서 정답을 도출하였다.")
6. **서답형 문항**: 정답이 "별지 참조" 등 수작업 채점이므로 구체적 답안 내용은 추정하지 않는다.
   학생이 해당 문항의 배점을 취득했다면 "해당 개념을 서술할 수 있다",
   미취득이면 "해당 서술에 이르지 못하였다"로 표현한다.
7. **등급·라벨 금지**: statement/note 본문에 "상", "중", "하", "A", "B", "C", "D", "E", "상 수준",
   "중 수준" 등 라벨 단어를 직접 쓰지 않는다. masteryLevel 필드에만 상/중/하를 기록한다.
8. **masteryLevel 판정** (필드값 전용): 배점 대비 득점률 — 상 80%+ / 중 50~79% / 하 50% 미만.
9. **standardsEvaluated[].note**: 한 줄이어도 구체 역량으로. 예시:
   - 적합: "DNA→RNA→단백질 흐름의 주요 단계를 서술할 수 있으나, 조절 유전자 사례의 연결은 미완결."
   - 부적합: "상 수준." / "대체로 이해함."
10. 응답은 반드시 JSON 스키마를 정확히 따른다 (추가 필드 금지, 누락 금지).`;

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
          statement: { type: 'string', description: '3~5문장 성취수준 진술문. 추상적 "상/중/하 수준" 표현이 아니라, 평가기준 A~E 스타일의 구체적 역량 기술(핵심 개념·단계·관계를 풀어 쓴 문장)을 사용해야 한다.' },
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
  lines.push('[요구사항]');
  lines.push('첨부된 출제원안지 PDF의 실제 문항·선지 내용과 위 성취기준 전문을 근거로, 위 학생의 영역별 성취수준을 JSON 스키마에 맞춰 분석하라.');
  lines.push('각 성취기준 문항에서 학생이 정답을 도출한 양상 × 난이도를 시스템 프롬프트의 "평가기준 A~E 해석 가이드"에 대입하여 판단한 뒤,');
  lines.push('시스템 프롬프트의 [진술문 구조] ①~⑤ 순서(역량 진술 → 근거 문항/선지 → 한계 → 난이도 참조 → 마무리)에 따라 statement를 3~5문장으로 작성한다.');
  lines.push('성취기준의 행위 동사("설명할 수 있다", "구분할 수 있다", "추론할 수 있다" 등)를 그대로 재사용하고,');
  lines.push('오답은 "틀렸다/부족하다" 대신 "정교하게 연결하는 데 이르지 못하였다" 류의 행동 기반 표현으로 기술한다.');
  lines.push('statement/note 본문에 "상", "중", "하", "A", "B"... 같은 라벨 단어는 절대 노출하지 않는다(masteryLevel 필드에만 상/중/하 기록).');
  lines.push('한 영역에 성취기준이 여러 개이면 성취기준 코드별로 구분하여 각각 최소 한 문장씩 다룬다.');
  lines.push('객관식 문항은 출제원안지 선지까지 확인하여 구체 개념(예: "규산염 사면체의 공유 결합 배열")을 인용한다.');

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
