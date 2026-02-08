# OmniRank Web

Next.js frontend for the staged OmniRank SOP backend.

## Workflow

The UI follows backend stages directly:

1. Upload CSV or load example dataset
2. Preview data
3. Run `/infer` (schema + format/quality validation)
4. User confirms schema/config in ranking preview bubble
5. Run `/run` for spectral ranking + deterministic SVG + report
6. Ask follow-up questions via `/question`

## Quote-Aware Report Loop

- Report renders backend markdown blocks (`data-omni-block-id`, `data-omni-kind`)
- User selects text and clicks **Quote**
- Frontend sends `quotes: QuotePayload[]` to `/question`
- Backend returns `used_citation_block_ids` for traceable evidence

## Scripts

```bash
npm run dev
npm run lint
npm run build
```

## Dev Startup (Hot Reload)

Frontend (this folder):

```bash
cd src/web
npm run dev
```

Backend (separate terminal):

```bash
cd src/api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

`npm run dev` and `uvicorn --reload` both auto-reload on code changes.
If you change `.env.local`, restart `npm run dev`.

## Environment

Frontend API base URL:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```
