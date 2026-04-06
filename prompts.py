# prompts.py — All prompt templates

def get_question_prompt(domain: str, difficulty: str, asked_questions: list[str], resume_context: str = "") -> str:
    if asked_questions:
        already_asked_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(asked_questions))
        avoid_section = f"""
CRITICAL RULE — STRICTLY AVOID REPETITION:
You MUST generate a completely new question that is DISTINCT in topic, wording, focus, and structure from ALL previously asked questions.

Do NOT repeat, rephrase, or closely resemble any previous question.
Previously asked:
{already_asked_text}
"""
    else:
        avoid_section = ""

    resume_section = ""
    if resume_context and resume_context.strip():
        resume_section = f"""
CANDIDATE BACKGROUND (from their resume):
{resume_context.strip()[:1500]}

Use this background to personalise the question:
- Reference specific technologies, tools, or projects they have listed
- Ask them to go deeper on something they claimed experience with
- Connect the question to their actual background where possible
- Do NOT ask about things completely absent from their resume
"""

    return f"""You are an experienced technical interviewer.
Generate exactly ONE new {difficulty} difficulty interview question for a {domain} position.

Rules:
- Output ONLY the question text — nothing else
- Clear, specific, answerable in 2–5 minutes verbally
- Engaging and relevant to real {domain} work
{resume_section}{avoid_section}
Domain: {domain}
Difficulty: {difficulty}"""

def get_evaluation_prompt(domain: str, difficulty: str, question: str, answer: str) -> str:
    return f"""You are a senior {domain} interviewer evaluating a candidate's answer.

Context:
- Domain: {domain}
- Difficulty: {difficulty}
- Question: {question}
- Answer: {answer}

Respond ONLY with valid JSON — no extra text:

{{
  "technical_score": <0-10>,
  "depth_score": <0-10>,
  "clarity_score": <0-10>,
  "overall_score": <0-10>,
  "short_feedback": "<2-3 sentence human-sounding reaction>",
  "ideal_answer_hint": "<1-2 sentence key missing point>"
}}

Scoring: 9-10 exceptional, 7-8 good, 5-6 average, 3-4 below avg, 0-2 poor."""

def get_reaction_prompt(domain: str, question: str, answer: str, feedback: str, overall_score: int) -> str:
    tone = "encouraging" if overall_score >= 7 else "constructive" if overall_score >= 4 else "direct but supportive"
    return f"""You just evaluated this answer (score {overall_score}/10, feedback: {feedback}).

Give a short 1-sentence spoken-style transition comment (8–20 words) before the next question.
Sound natural and conversational.

Examples:
- "Good grasp of the basics — let's see how you handle scaling."
- "You missed a key edge case. Next one will test that."

Tone: {tone}
Output ONLY the sentence."""

def get_followup_prompt(domain: str, question: str, answer: str, feedback: str, overall_score: int) -> str:
    tone = "curious and probing" if overall_score >= 6 else "gently corrective" if overall_score >= 3 else "supportive but pointing out gaps"

    return f"""You are the same senior {domain} interviewer.

Previous question: {question}
Candidate answer: {answer}
Your feedback: {feedback}
Score: {overall_score}/10

Now generate ONE short follow-up / probing / clarifying question (1 sentence) to dig deeper or address a gap.

Rules:
- Output ONLY the follow-up question text — nothing else
- Keep it natural, conversational, 15–40 words
- Focus on clarifying, challenging, or extending the previous answer
- Do NOT repeat the original question
- Do NOT ask something completely unrelated

Tone: {tone}
Example follow-ups:
- "Can you walk me through how you'd handle the case when the cache is full?"
- "Interesting choice — why did you prefer composition over inheritance here?"
- "You mentioned scalability — what trade-offs did you consider?"

Output ONLY the question."""

def get_final_report_prompt(domain: str, difficulty: str, qa_history: list[dict]) -> str:
    history_text = ""
    for i, item in enumerate(qa_history):
        scores = item.get('scores', {})
        history_text += f"Q{i+1}: {item.get('question', '')}\nAnswer: {item.get('answer', '')}\nScores: {scores}\n\n"

    return f"""You are writing a final performance review.

Details:
- Domain: {domain}
- Difficulty: {difficulty}
- Questions: {len(qa_history)}

Session:
{history_text}

Respond ONLY with JSON:

{{
  "average_score": <float, 1 decimal>,
  "readiness_level": "<Needs More Preparation / Getting There / Interview Ready / Strong Candidate>",
  "strongest_areas": ["..."],
  "weak_areas": ["..."],
  "overall_summary": "<3-4 sentences>",
  "top_improvements": ["...","...","..."],
  "encouragement": "<1 motivating sentence>"
}}"""

def get_confidence_prompt(question: str, answer: str) -> str:
    word_count = len(answer.strip().split()) if answer else 0
    json_schema = """
{
  "certainty": <1-10>,
  "structure": <1-10>,
  "assertiveness": <1-10>,
  "vocabulary": <1-10>,
  "overall": <1-10, weighted average>,
  "coaching_tip": "<one encouraging tip to go from good to great, max 15 words>"
}"""
    return f"""You are analysing a spoken job interview answer for confidence and communication quality.

Answer word count: {word_count} words

SCORING RULES — follow these strictly:

SHORT ANSWERS (under 15 words): Score 1-3 across all dimensions. These answers are too brief to demonstrate knowledge or confidence. A 1-2 word answer must score 1-2 overall.

MEDIUM ANSWERS (15-40 words): Score 4-6. Some effort shown but not enough depth.

FULL ANSWERS (40+ words): Score based on quality:
- 8-9 = normal good spoken answer, candidate knows their stuff — this is the DEFAULT for genuine 40+ word attempts
- 6-7 = slightly hesitant or disorganised but solid
- 4-5 = noticeably weak, lots of hedging or poor structure
- 1-3 = incoherent, rambling, or extremely unconfident

IMPORTANT:
- Occasional "um", "like", "you know" are normal in speech — do not penalise them
- This is transcribed speech, not written text — judge accordingly
- If the answer is just 1-5 words, overall MUST be 1 or 2 — no exceptions

Question: {question}
Answer: {answer}

Score these 4 dimensions (follow the word count rules above):

1. CERTAINTY — Does the candidate sound like they know what they're talking about?
2. STRUCTURE — Does the answer have a logical flow?
3. ASSERTIVENESS — Does the candidate own their answer?
4. VOCABULARY — Is the language precise and domain-appropriate?

Coaching tip: ONE short encouraging tip (max 15 words). If answer is very short, tell them to elaborate more.

Respond ONLY with valid JSON — no extra text:
{json_schema}"""