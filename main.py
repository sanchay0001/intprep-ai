# ══════════════════════════════════════════════════════════════════════════════
# main.py — FastAPI Application Entry Point
# ══════════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ai_engine import (
    generate_question,
    evaluate_answer,
    generate_final_report,
    generate_reaction,
    generate_followup,
    cleanup_transcript,
    analyze_confidence
)

app = FastAPI(
    title="AI Interview Platform",
    description="A mock interview simulator powered by Groq LLM",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class StartInterviewRequest(BaseModel):
    domain: str
    difficulty: str
    num_questions: int

class NextQuestionRequest(BaseModel):
    domain: str
    difficulty: str
    asked_questions: list[str]
    resume_context: str = ""     # optional — empty string means no resume

class EvaluateAnswerRequest(BaseModel):
    domain: str
    difficulty: str
    question: str
    answer: str

class FinalReportRequest(BaseModel):
    domain: str
    difficulty: str
    qa_history: list[dict]

class CleanupTranscriptRequest(BaseModel):   # ← new
    raw_transcript: str
    domain: str
    question: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate-question")
async def get_question(data: NextQuestionRequest):
    try:
        question = generate_question(
            domain=data.domain,
            difficulty=data.difficulty,
            asked_questions=data.asked_questions,
            resume_context=data.resume_context
        )
        return {"question": question}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to generate question: {str(e)}"})

@app.post("/evaluate-answer")
async def evaluate(data: EvaluateAnswerRequest):
    try:
        result = evaluate_answer(
            domain=data.domain,
            difficulty=data.difficulty,
            question=data.question,
            answer=data.answer
        )
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to evaluate answer: {str(e)}"})

@app.post("/final-report")
async def final_report(data: FinalReportRequest):
    try:
        report = generate_final_report(
            domain=data.domain,
            difficulty=data.difficulty,
            qa_history=data.qa_history
        )
        return report
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to generate report: {str(e)}"})

@app.post("/generate-reaction")
async def get_reaction(data: dict):
    try:
        reaction = generate_reaction(
            domain=data["domain"],
            question=data["question"],
            answer=data["answer"],
            feedback=data["short_feedback"],
            overall_score=data["overall_score"]
        )
        return {"reaction": reaction}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/generate-followup")
async def get_followup(data: dict):
    try:
        followup = generate_followup(
            domain=data["domain"],
            question=data["question"],
            answer=data["answer"],
            feedback=data["short_feedback"],
            overall_score=data["overall_score"]
        )
        return {"followup": followup}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/cleanup-transcript")                  # ← new endpoint
async def cleanup(data: CleanupTranscriptRequest):
    try:
        cleaned = cleanup_transcript(
            raw_transcript=data.raw_transcript,
            domain=data.domain,
            question=data.question
        )
        return {"cleaned_transcript": cleaned}
    except Exception as e:
        # Always return something usable — fall back to raw if cleanup fails
        return {"cleaned_transcript": data.raw_transcript, "warning": str(e)}

class ConfidenceRequest(BaseModel):
    question: str
    answer: str

@app.post("/analyze-confidence")
async def confidence(data: ConfidenceRequest):
    try:
        result = analyze_confidence(
            question=data.question,
            answer=data.answer
        )
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "AI Interview Platform is running"}