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
    get_followup_prompt
)

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

def _call_groq(prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
    response = client.chat.completions.create(
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

def generate_question(domain: str, difficulty: str, asked_questions: list[str]) -> str:
    prompt = get_question_prompt(domain, difficulty, asked_questions)
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
    parsed = _parse_json_response(raw_response)
    return parsed

def generate_reaction(domain: str, question: str, answer: str, feedback: str, overall_score: int) -> str:
    prompt = get_reaction_prompt(domain, question, answer, feedback, overall_score)
    raw = _call_groq(prompt, max_tokens=80, temperature=0.5)
    return raw.strip().strip('"').strip("'")

def generate_followup(domain: str, question: str, answer: str, feedback: str, overall_score: int) -> str:
    prompt = get_followup_prompt(domain, question, answer, feedback, overall_score)
    raw = _call_groq(prompt, max_tokens=100, temperature=0.6)
    return raw.strip().strip('"').strip("'")