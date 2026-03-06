# prompts.py — All prompt templates

def get_question_prompt(domain: str, difficulty: str, asked_questions: list[str]) -> str:
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

    return f"""You are an experienced technical interviewer.
Generate exactly ONE new {difficulty} difficulty interview question for a {domain} position.

Rules:
- Output ONLY the question text — nothing else
- Clear, specific, answerable in 2–5 minutes verbally
- Engaging and relevant to real {domain} work
{avoid_section}
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