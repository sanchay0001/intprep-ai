# ══════════════════════════════════════════════════════════════════════════════
# ai_engine.py — Core AI logic for the Interview Platform
# ══════════════════════════════════════════════════════════════════════════════

import os
import json
import re

from groq import Groq
from dotenv import load_dotenv
from prompts import (
    get_question_prompt,
    get_evaluation_prompt,
    get_final_report_prompt,
    get_reaction_prompt,
    get_followup_prompt,
    get_confidence_prompt
)

load_dotenv()

# Primary Groq client — all interview logic
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

# Secondary Groq client — transcript cleanup (separate key for free credits)
# Falls back to primary key if GROQ_API_KEY_2 is not set
_cleanup_key = os.getenv("GROQ_API_KEY_2") or os.getenv("GROQ_API_KEY")
cleanup_client = Groq(api_key=_cleanup_key)
CLEANUP_MODEL = "llama-3.1-8b-instant"   # smallest/fastest Groq model — perfect for cleanup

def _call_groq(prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

def _parse_json_response(raw_text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
    cleaned = cleaned.replace("```", "").strip()
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(0)
    return json.loads(cleaned)

def generate_question(domain: str, difficulty: str, asked_questions: list[str], resume_context: str = "") -> str:
    prompt = get_question_prompt(domain, difficulty, asked_questions, resume_context)
    question_text = _call_groq(prompt, max_tokens=200, temperature=0.4)
    return question_text.strip().strip('"').strip("'")

def evaluate_answer(domain: str, difficulty: str, question: str, answer: str) -> dict:
    if not answer or answer.strip() == "":
        return {
            "technical_score": 0,
            "depth_score": 0,
            "clarity_score": 0,
            "overall_score": 0,
            "short_feedback": "No answer was provided. Always attempt an answer in a real interview.",
            "ideal_answer_hint": "Explain your reasoning even if unsure."
        }
    prompt = get_evaluation_prompt(domain, difficulty, question, answer)
    raw_response = _call_groq(prompt, max_tokens=600)
    return _parse_json_response(raw_response)

def generate_final_report(domain: str, difficulty: str, qa_history: list[dict]) -> dict:
    prompt = get_final_report_prompt(domain, difficulty, qa_history)
    raw_response = _call_groq(prompt, max_tokens=900)
    return _parse_json_response(raw_response)

def generate_reaction(domain: str, question: str, answer: str, feedback: str, overall_score: int) -> str:
    prompt = get_reaction_prompt(domain, question, answer, feedback, overall_score)
    raw = _call_groq(prompt, max_tokens=80, temperature=0.5)
    return raw.strip().strip('"').strip("'")

def generate_followup(domain: str, question: str, answer: str, feedback: str, overall_score: int) -> str:
    prompt = get_followup_prompt(domain, question, answer, feedback, overall_score)
    raw = _call_groq(prompt, max_tokens=100, temperature=0.6)
    return raw.strip().strip('"').strip("'")

def cleanup_transcript(raw_transcript: str, domain: str, question: str) -> str:
    """
    Uses a secondary Groq key + lightweight Llama model to clean up garbled
    speech-to-text output before evaluation.
    Fixes mishearing, removes filler words, repairs broken sentences.
    Never adds content the user didn't say — only cleans what's there.
    Falls back to raw transcript on any failure.
    """
    if not raw_transcript or len(raw_transcript.strip()) < 15:
        return raw_transcript   # too short to bother cleaning

    prompt = f"""You are fixing garbled speech-to-text output from a voice interview.

Domain: {domain}
Question asked: {question}

Raw transcript (may contain mishearing, filler words, broken restarts):
\"\"\"{raw_transcript}\"\"\"

Your job:
- Fix obviously misheard technical terms (e.g. "neural nett work" → "neural network")
- Remove filler words: um, uh, like, you know, basically, literally, right, so so
- Fix broken mid-sentence restarts (e.g. "I think the the way to" → "I think the way to")
- Fix obvious punctuation and sentence flow
- Preserve ALL the candidate's actual ideas, content, and technical points exactly
- Do NOT add any information, examples, or explanations the candidate didn't say
- Do NOT improve the quality of the answer — only fix transcription artifacts

Output ONLY the cleaned transcript text. No explanations, no labels, no quotes."""

    try:
        response = cleanup_client.chat.completions.create(
            model=CLEANUP_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1,   # low temp — we want deterministic cleanup, not creative
        )
        cleaned = response.choices[0].message.content.strip()
        cleaned = cleaned.strip('"').strip("'")   # strip any quotes the model adds
        # Sanity check: if result is suspiciously short, fall back to raw
        if not cleaned or len(cleaned) < len(raw_transcript) * 0.3:
            return raw_transcript
        return cleaned
    except Exception as e:
        # Never block evaluation due to cleanup failure
        print(f"[cleanup_transcript] Groq cleanup failed, using raw: {e}")
        return raw_transcript

def analyze_confidence(question: str, answer: str) -> dict:
    """
    Analyses a transcript for confidence signals.
    Uses the cleanup Groq client (secondary key) to save primary key quota.
    Falls back to neutral scores on any failure — never blocks evaluation.
    """
    if not answer or len(answer.strip()) < 20:
        return {
            "certainty": 5, "structure": 5, "assertiveness": 5,
            "vocabulary": 5, "overall": 5,
            "coaching_tip": "Give a fuller answer to get confidence feedback."
        }

    prompt = get_confidence_prompt(question, answer)
    try:
        response = cleanup_client.chat.completions.create(
            model=CLEANUP_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        return _parse_json_response(raw)
    except Exception as e:
        print(f"[analyze_confidence] Failed, returning neutral: {e}")
        return {
            "certainty": 5, "structure": 5, "assertiveness": 5,
            "vocabulary": 5, "overall": 5,
            "coaching_tip": "Keep practising to build confidence."
        }