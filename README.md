# Skill Gap Analyser

Connects the RAG job-retrieval engine to the Gemini career-advisor into one working pipeline.

    rag_engine.py       -> Retriever: loads LinkedIn_Jobs_Data_India.csv, embeds it, retrieves
                           matching jobs for a (state, job_title), computes the skill gap
    career_advisor.py   -> generate_career_recommendations(): sends that gap to Gemini,
                           returns a structured learning plan (schema-constrained JSON)
    main.py             -> orchestrator: bridges the two, exposes manual-skill-input CLI
                           and a combined FastAPI endpoint

Resume parsing is NOT included yet -- skills are entered manually (a list of strings).
That's the one open slot to wire in later (feed extracted skills into `user_skills` /
`skills` instead of typing them).

## Setup

    pip install -r requirements.txt
    set GOOGLE_API_KEY=your_key_here      # or put it in a .env file in this folder

## 1. Build the index (once)

    python main.py build --csv "C:\Users\aronm\Downloads\archive\LinkedIn_Jobs_Data_India.csv"

## 2. Run

Manual terminal mode (no server):

    python main.py interactive

You'll be prompted for state, target job title, and a comma-separated skill list --
it prints the retrieved jobs' skill gap and the Gemini learning plan.

Or as an API:

    uvicorn main:app --reload

    POST /api/v1/analyze
    {
      "state": "Karnataka",
      "job_title": "Backend Python Developer",
      "skills": ["Python", "SQL", "Git"],
      "top_k": 8
    }

Returns `{ retrieval: {...jobs, skill_gap...}, career_plan: {...Gemini plan...} }`.
If there's no missing skill against the retrieved postings, `career_plan` is `null`
and `note` explains why -- Gemini isn't called in that case.
