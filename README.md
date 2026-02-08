# OmniRank

OmniRank is a monorepo implementing a strict single-agent SOP for spectral ranking inference.

## Architecture

- Backend: `src/api` (FastAPI + single OmniRank agent + fixed 10-tool registry)
- Frontend: `src/web` (Next.js + staged HTTP workflow + quote-aware report Q&A)
- Spectral engine: `src/spectral_ranking/spectral_ranking.R`
- Shared contracts: `shared/types/api.ts`

The production pipeline is fixed:

1. `read_data_file`
2. `infer_semantic_schema`
3. `validate_data_format`
4. (`preprocess_data` + re-validate loop when fixable)
5. `validate_data_quality`
6. `request_user_confirmation`
7. `execute_spectral_ranking`
8. `generate_visualizations`
9. `generate_report`
10. `answer_question` (quote-aware follow-up loop)

## Default Model

- `OPENAI_MODEL=gpt-5-mini`
- No silent fallback model is used.

## HTTP API

Main staged endpoints:

- `POST /api/upload`
- `POST /api/upload/example/{example_id}`
- `GET /api/preview/{session_id}`
- `POST /api/sessions/{session_id}/infer`
- `POST /api/sessions/{session_id}/confirm`
- `POST /api/sessions/{session_id}/run`
- `POST /api/sessions/{session_id}/question`
- `GET /api/sessions/{session_id}/artifacts/{artifact_id}`
- `GET /api/sessions/{session_id}`
- `DELETE /api/sessions/{session_id}`

## Local Development (Auto Reload)

Prerequisites:

- Python 3.11+
- Node.js 20+
- R 4.0+ (`Rscript` available in PATH)

Use two terminals.

Terminal 1 (backend, auto reload on Python file changes):

```bash
cd src/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2 (frontend, hot reload on TS/TSX/CSS changes):

```bash
cd src/web
npm install
npm run dev
```

Notes:

- Backend URL must match `NEXT_PUBLIC_API_URL` in `src/web/.env.local` (default: `http://localhost:8000`).
- Changing `.env` or `.env.local` requires process restart; code changes do not.

## Validation Commands

Backend tests:

```bash
pytest -q
```

Frontend lint/build:

```bash
cd src/web
npm run lint
npm run build
```

## Notes

- `Spectral_Ranking/` is read-only reference material and is not part of runtime code.
- New runtime path uses HTTP stages; legacy multi-agent modules are archived under `archived/`.
