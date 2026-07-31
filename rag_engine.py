"""Placement Support RAG engine — retrieval + skill-gap analysis.

Import and use as a library:

    from rag_engine import Retriever, SearchRequest

    retriever = Retriever()
    retriever.build("LinkedIn_Jobs_Data_India.csv")   # first time only
    result = retriever.search(SearchRequest(
        state="Karnataka",
        job_title="Backend Python Developer",
        user_skills=["Python", "SQL", "Git"],
    ))

`result["skill_gap"]` is what feeds into career_advisor.py.

Install:
  pip install sentence-transformers faiss-cpu numpy pandas pydantic rapidfuzz
"""

from __future__ import annotations

import html
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Annotated, Iterable

import faiss
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("PLACEMENT_RAG_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_DIR = Path(os.getenv("PLACEMENT_RAG_INDEX_DIR", "data/placement_rag_index"))
REQUIRED_COLUMNS = {"title", "description", "state", "city", "companyName"}


class SearchRequest(BaseModel):
    state: Annotated[str, Field(min_length=2, max_length=100)]
    job_title: Annotated[str, Field(min_length=2, max_length=200)]
    user_skills: list[str] = Field(default_factory=list, max_length=250)
    resume_text: str | None = Field(default=None, max_length=100_000)
    top_k: Annotated[int, Field(default=8, ge=1, le=25)] = 8

    @field_validator("state", "job_title")
    @classmethod
    def non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


def key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).casefold().strip())


def normalise_skill(value: str) -> str:
    value = html.unescape(str(value)).lower().strip()
    return re.sub(r"\s+", " ", value).strip(" .,:;|/\\")


def unique_skills(values: Iterable[str]) -> list[str]:
    result, seen = [], set()
    for value in values:
        skill = normalise_skill(value)
        if 2 <= len(skill) <= 60 and skill not in seen:
            result.append(skill)
            seen.add(skill)
    return result


def skills_from_description(description: str) -> list[str]:
    """Extract the dataset's structured `Skill: ...; Exp:` segment."""
    if not isinstance(description, str):
        return []
    found = re.search(
        r"\bskills?\s*:\s*(.{1,250}?)(?:;\s*(?:exp(?:erience)?|job\s+description)\b|\n|$)",
        description,
        re.I | re.S,
    )
    return unique_skills(re.split(r"[,|•]", found.group(1))) if found else []


def document(job: dict) -> str:
    return "\n".join((
        f"Job title: {job['title']}", f"Company: {job['company']}",
        f"Location: {job['city']}, {job['state']}",
        f"Experience: {job['experience_level']}", f"Description: {job['description']}",
        "Required skills: " + ", ".join(job["skills"]),
    ))


class Retriever:
    """Hybrid local RAG: hard location/title filters precede semantic ranking."""

    def __init__(self, index_dir: Path = INDEX_DIR, model_name: str = MODEL_NAME):
        self.index_dir, self.model_name = index_dir, model_name
        self.model: SentenceTransformer | None = None
        self.records: list[dict] = []
        self.embeddings: np.ndarray | None = None

    def embedding_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def build(self, csv_path: str) -> dict:
        frame = pd.read_csv(csv_path)
        frame = frame.loc[:, ~frame.columns.astype(str).str.match(r"^(Unnamed: \d+|$)")]
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        records = []
        for _, row in frame.fillna("").iterrows():
            title, description, state = (str(row[n]).strip() for n in ("title", "description", "state"))
            if title and description and state:
                records.append({
                    "job_id": str(row.get("id", "")), "title": title, "description": description,
                    "state": state, "city": str(row.get("city", "")).strip(),
                    "company": str(row.get("companyName", "")).strip(),
                    "experience_level": str(row.get("experienceLevel", "")).strip(),
                    "contract_type": str(row.get("contractType", "")).strip(),
                    "skills": skills_from_description(description),
                })
        if not records:
            raise ValueError("No indexable job rows found")
        vectors = np.asarray(
            self.embedding_model().encode(
                [document(x) for x in records], normalize_embeddings=True, show_progress_bar=True
            ),
            dtype="float32",
        )
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_dir / "jobs.faiss"))
        np.save(self.index_dir / "embeddings.npy", vectors)
        (self.index_dir / "jobs.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        (self.index_dir / "manifest.json").write_text(
            json.dumps({"model": self.model_name, "rows": len(records)}), encoding="utf-8"
        )
        self.records, self.embeddings = records, vectors
        return {"indexed_jobs": len(records), "model": self.model_name, "index_dir": str(self.index_dir)}

    def load(self) -> None:
        manifest = self.index_dir / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError("Index missing. Call retriever.build(csv_path) first.")
        self.model_name = json.loads(manifest.read_text(encoding="utf-8"))["model"]
        self.records = json.loads((self.index_dir / "jobs.json").read_text(encoding="utf-8"))
        self.embeddings = np.load(self.index_dir / "embeddings.npy").astype("float32")

    def is_ready(self) -> bool:
        return self.embeddings is not None or (self.index_dir / "manifest.json").exists()

    def search(self, payload: SearchRequest) -> dict:
        if self.embeddings is None:
            self.load()
        assert self.embeddings is not None
        exact_state = [i for i, job in enumerate(self.records) if key(job["state"]) == key(payload.state)]
        if not exact_state:
            choices = sorted({job["state"] for job in self.records})[:20]
            raise ValueError(f"No exact state match for '{payload.state}'. Available values include: {choices}")
        related_title = [
            i for i in exact_state
            if fuzz.token_set_ratio(key(payload.job_title), key(self.records[i]["title"])) >= 45
        ]
        candidate_ids = related_title or exact_state
        filter_mode = "exact_state+related_title" if related_title else "exact_state_only_no_related_title"
        query = f"Target job title: {payload.job_title}. Relevant job requirements and skills."
        vector = np.asarray(self.embedding_model().encode([query], normalize_embeddings=True), dtype="float32")[0]
        scores = self.embeddings[candidate_ids] @ vector
        ranked = sorted(zip(candidate_ids, scores.tolist()), key=lambda x: x[1], reverse=True)[:payload.top_k]
        jobs = []
        for job_id, score in ranked:
            job = self.records[job_id]
            result = {
                field: job[field]
                for field in ("job_id", "title", "company", "city", "state", "experience_level", "contract_type", "skills")
            }
            result.update(semantic_score=round(float(score), 4), description_excerpt=job["description"][:700])
            jobs.append(result)
        required = self.required_skills(jobs)
        candidate_skills = unique_skills(payload.user_skills)
        if payload.resume_text:
            text = normalise_skill(payload.resume_text)
            vocabulary = [x["skill"] for x in required]
            for skill in vocabulary:
                if len(skill) >= 3 and re.search(rf"(?<![a-z0-9+#.-]){re.escape(skill)}(?![a-z0-9+#.-])", text, re.I):
                    candidate_skills.append(skill)
        candidate_skills = unique_skills(candidate_skills)
        known = {normalise_skill(x) for x in candidate_skills}
        matched = [x for x in required if normalise_skill(x["skill"]) in known]
        missing = [x for x in required if normalise_skill(x["skill"]) not in known]
        gap = {
            "required_skills": required,
            "matched_skills": matched,
            "missing_skills": missing,
            "candidate_skills_used": candidate_skills,
            "coverage_percent": round(100 * len(matched) / len(required), 1) if required else 0.0,
        }
        return {
            "query": {"state": payload.state, "job_title": payload.job_title},
            "retrieval": {"filter_mode": filter_mode, "candidate_count": len(candidate_ids), "returned_count": len(jobs)},
            "jobs": jobs,
            "skill_gap": gap,
            "llm_context": self.llm_context(payload.state, payload.job_title, jobs, gap),
        }

    @staticmethod
    def required_skills(jobs: list[dict], limit: int = 20) -> list[dict]:
        scores, evidence = Counter(), {}
        for job in jobs:
            for skill in job["skills"]:
                scores[skill] += max(job["semantic_score"], 0.05)
                evidence.setdefault(skill, []).append(job["title"])
        return [
            {"skill": skill, "importance": round(score, 4), "seen_in_titles": list(dict.fromkeys(evidence[skill]))[:3]}
            for skill, score in scores.most_common(limit)
        ]

    @staticmethod
    def llm_context(state: str, title: str, jobs: list[dict], gap: dict) -> str:
        lines = [
            "PLACEMENT RETRIEVAL CONTEXT", f"Target: {title} | Exact state: {state}",
            "Use only the evidence below; do not invent job requirements.", "", "Skill-gap evidence:",
        ]
        lines += [
            "- Matched: " + (", ".join(x["skill"] for x in gap["matched_skills"]) or "none"),
            "- Missing: " + (", ".join(x["skill"] for x in gap["missing_skills"]) or "none"),
            f"- Coverage: {gap['coverage_percent']}% of retrieved required skills", "", "Retrieved job evidence:",
        ]
        lines += [
            f"- {j['title']} | {j['company']} | {j['city']}, {j['state']} | skills: {', '.join(j['skills']) or 'not explicitly listed'}"
            for j in jobs
        ]
        return "\n".join(lines)
