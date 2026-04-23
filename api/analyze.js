// ============================================================================
// AI 성취수준 분석 API (Gemini 2.5 Flash 멀티모달)
// ----------------------------------------------------------------------------
// 환경 변수 (Vercel 대시보드 > Settings > Environment Variables):
//   GEMINI_API_KEY  (필수)  Gemini API 키
//   GEMINI_MODEL    (선택)  기본값: gemini-2.5-flash
// ============================================================================

const SYSTEM_PROMPT = `너는 고등학교 과학 평가 분석 전문가다.
학생 한 명의 시험 응답을 분석하여 내용 영역별 성취수준을 객관적으로 진술한다.

[엄격 규칙]
1. 성취기준 코드(예: [10통과1-01-02])를 반드시 인용한다.
2. 학생을 격려·비난하는 표현을 쓰지 않는다. 객관적·중립적 서술만.
3. "~할 수 있다", "~에 대한 이해가 부족하다", "~문항에서 정답을 도출하지 못했다" 등 학습과학 용어를 사용한다.
4. 각 내용 영역 진술문은 3~5문장이고 구체적 근거(어떤 성취기준 문항에서 맞고/틀렸는지)를 포함한다.
5. 성취수준은 배점 대비 득점률로 판정:
   - 상: 80% 이상
   - 중: 50~79%
   - 하: 50% 미만
6. 서답형 문항은 정답 여부가 "별지 참조"/수작업 채점이므로, 배점 정보만 참고하고 구체적 옳고그름은 판단하지 않는다.
7. 출력은 반드시 아래 JSON 스키마를 따른다.

[출력 JSON 스키마]
{
  "areas": [
    {
      "areaName": "과학의 기초",
      "totalPoints": number,
      "scoredPoints": number,
      "attainmentPercent": number,
      "masteryLevel": "상" | "중" | "하",
      "statement": "객관적 성취수준 진술문 (3~5문장)",
      "standardsEvaluated": [
        { "code": "10통과1-01-02", "hits": "3/4", "note": "짧은 평가" }
      ]
    }
  ],
  "overallNote": "전 영역 종합 1~2문장"
}`;

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', req.headers.origin || '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') { res.status(204).end(); return; }
  if (req.method !== 'POST') { res.status(405).json({ error: 'Method not allowed' }); return; }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    res.status(500).json({ error: '서버에 GEMINI_API_KEY가 설정되지 않았습니다.' });
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

  const model = process.env.GEMINI_MODEL || 'gemini-2.5-flash';

  // 사용자 프롬프트 구성
  const userText = buildUserPrompt(itemInfo, student);

  // Gemini 요청 구성 (멀티모달: PDF + 텍스트)
  const parts = [{ text: userText }];
  if (testPaperPdf && typeof testPaperPdf === 'string' && testPaperPdf.length > 100) {
    parts.push({
      inlineData: {
        mimeType: 'application/pdf',
        data: testPaperPdf
      }
    });
  }

  try {
    const geminiUrl = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(apiKey)}`;

    const geminiResp = await fetch(geminiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
        contents: [{ role: 'user', parts }],
        generationConfig: {
          responseMimeType: 'application/json',
          temperature: 0.3,
          maxOutputTokens: 4096
        }
      })
    });

    if (!geminiResp.ok) {
      const errText = await geminiResp.text();
      res.status(geminiResp.status).json({
        error: `Gemini API 오류 (${geminiResp.status}): ${errText.slice(0, 300)}`
      });
      return;
    }

    const data = await geminiResp.json();
    const raw = data.candidates?.[0]?.content?.parts?.[0]?.text || '';

    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (_) {
      res.status(500).json({ error: 'AI 응답을 JSON으로 파싱할 수 없습니다.', raw: raw.slice(0, 500) });
      return;
    }

    res.status(200).json(parsed);
  } catch (err) {
    console.error('Analyze error:', err);
    res.status(500).json({ error: err.message || '서버 내부 오류' });
  }
}

function buildUserPrompt(itemInfo, student) {
  // 학생의 응답 정리 (익명화된 ID만 사용)
  const safeId = (student.id || student.anonId || 'student').toString().slice(0, 30);

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
  lines.push('첨부된 출제원안지 PDF의 실제 문항 내용을 참고하여, 위 학생의 영역별 성취수준을 위 JSON 스키마에 맞춰 분석하라.');
  lines.push('성취기준 코드별로 어떤 개념이 강점/약점인지 구체적으로 서술하되, 학생을 평가(격려/비난)하지 말고 객관 사실만 서술한다.');

  return lines.join('\n');
}

export const config = {
  api: {
    bodyParser: { sizeLimit: '12mb' },
    responseLimit: false
  }
};
