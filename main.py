"""
Skill Gap Analyser — combined pipeline.

    resume/manual skills  --->  rag_engine.Retriever.search()   (job dataset + skill gap)
                                          |
                                          v
                         career_advisor.generate_career_recommendations()  (Gemini plan)
                                          |
                                          v
                              final structured learning plan

Resume extraction is NOT wired in yet — skills are entered manually for now.

Setup (Windows paths shown, adjust for your OS):

    pip install fastapi "uvicorn[standard]" sentence-transformers faiss-cpu numpy pandas ^
                pydantic pydantic-settings rapidfuzz google-genai python-dotenv

    set GOOGLE_API_KEY=your_key_here          (or put it in a .env file in this folder)

Build the job index once:

    python main.py build --csv C:\\Users\\aronm\\Downloads\\archive\\LinkedIn_Jobs_Data_India.csv

Then either:

    python main.py interactive          # manual terminal input, no server needed
    uvicorn main:app --reload           # or run the combined API
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from career_advisor import (
    GeminiGenerationError,
    RecommendationResponse,
    SkillGapRequest,
    generate_career_recommendations,
    get_settings,
)
from rag_engine import Retriever, SearchRequest

INDEX_DIR = Path(os.getenv("PLACEMENT_RAG_INDEX_DIR", "data/placement_rag_index"))

retriever = Retriever(INDEX_DIR)


# ==========================================================================
# Bridge: rag_engine's skill_gap output -> career_advisor's SkillGapRequest
# ==========================================================================

def to_skill_gap_request(job_title: str, skill_gap: dict) -> SkillGapRequest | None:
    """Returns None when there's nothing missing -- no point calling the LLM."""
    missing = [x["skill"] for x in skill_gap["missing_skills"]]
    if not missing:
        return None
    return SkillGapRequest(
        target_role=job_title,
        existing_skills=skill_gap["candidate_skills_used"],
        missing_skills=missing,
    )


async def run_pipeline(state: str, job_title: str, user_skills: list[str], top_k: int = 8) -> dict:
    """The full pipeline: retrieve jobs + compute gap, then get a Gemini learning plan."""
    search_result = retriever.search(
        SearchRequest(state=state, job_title=job_title, user_skills=user_skills, top_k=top_k)
    )
    gap_request = to_skill_gap_request(job_title, search_result["skill_gap"])

    if gap_request is None:
        return {
            "retrieval": search_result,
            "career_plan": None,
            "note": "No missing skills detected against the retrieved postings -- full coverage, so no plan was generated.",
        }

    plan = await generate_career_recommendations(gap_request, get_settings())
    return {"retrieval": search_result, "career_plan": plan.model_dump(), "note": None}


# ==========================================================================
# FastAPI app -- combined endpoint
# ==========================================================================

class AnalyzeRequest(BaseModel):
    state: str
    job_title: str
    skills: list[str] = Field(default_factory=list, description="Manually entered skill set for now.")
    top_k: int = 8


app = FastAPI(title="Skill Gap Analyser", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"rag_index_ready": retriever.is_ready()}


@app.post("/api/v1/index")
def build_index(csv_path: str) -> dict:
    try:
        return retriever.build(csv_path)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/analyze")
async def analyze(payload: AnalyzeRequest) -> dict:
    """One call: retrieve relevant jobs, compute skill gap, get a Gemini learning plan."""
    try:
        return await run_pipeline(payload.state, payload.job_title, payload.skills, payload.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GeminiGenerationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ==========================================================================
# CLI: build index / interactive manual-skill-input mode
# ==========================================================================

def interactive_loop() -> None:
    print("Skill Gap Analyser -- manual input mode")
    print("Press Ctrl+C to exit.\n")
    while True:
        try:
            state = input("State (exact, e.g. Karnataka): ").strip()
            job_title = input("Target job title: ").strip()
            skills_raw = input("Your skills, comma-separated (manual entry): ").strip()
            skills = [s.strip() for s in skills_raw.split(",") if s.strip()]

            print("\nRetrieving matching jobs and computing skill gap...\n")
            result = asyncio.run(run_pipeline(state, job_title, skills))

            gap = result["retrieval"]["skill_gap"]
            print(f"Coverage: {gap['coverage_percent']}%")
            print("Matched skills: " + (", ".join(x["skill"] for x in gap["matched_skills"]) or "none"))
            print("Missing skills: " + (", ".join(x["skill"] for x in gap["missing_skills"]) or "none"))
            print()

            if result["career_plan"] is None:
                print(result["note"])
            else:
                print("=== Gemini Career Plan ===")
                print(json.dumps(result["career_plan"], indent=2, ensure_ascii=False))

            print("\n" + "=" * 72 + "\n")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        except (OSError, ValueError) as exc:
            print(f"\nError: {exc}\n")
        except GeminiGenerationError as exc:
            print(f"\nGemini error: {exc}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skill Gap Analyser -- build index or run interactively")
    parser.add_argument("command", choices=["build", "interactive"])
    parser.add_argument("--csv", default=os.getenv("PLACEMENT_RAG_CSV"))
    args = parser.parse_args()

    if args.command == "build":
        if not args.csv:
            parser.error("build requires --csv <path-to-LinkedIn_Jobs_Data_India.csv>")
        print(retriever.build(args.csv))
    else:
        interactive_loop()
