# OmniRank

> LLM Agent Platform for Statistically Rigorous Ranking Inference from Arbitrary Multiway Comparisons

OmniRank is a web-based agentic framework that democratizes access to spectral ranking inferences. The platform combines LLM reasoning capabilities with mathematically rigorous spectral ranking computation.

## Project Structure

```
OmniRank/
├── src/
│   ├── web/               # Next.js 15 Frontend
│   ├── api/               # FastAPI Backend
│   └── spectral_ranking/  # R Scripts (Spectral Engine)
├── shared/
│   └── types/             # Shared type definitions
├── data/
│   └── examples/          # Example datasets
├── docs/
│   └── project/           # Project documentation
└── .agent/
    ├── blueprints/        # Development blueprints
    ├── skills/            # Agent skills
    └── rules.md           # Project rules
```

## Tech Stack

### Frontend
- Next.js 15 (App Router)
- React 18
- TypeScript
- Tailwind CSS + shadcn/ui
- Framer Motion
- Recharts
- react-force-graph

### Backend
- FastAPI
- LangGraph
- Pydantic
- OpenAI GPT

### Spectral Engine
- R >= 4.0

## Getting Started

### Prerequisites
- Node.js >= 20
- **Python >= 3.11** (required for langgraph)
- R >= 4.0

### Frontend Setup

```bash
cd src/web
npm install
npm run dev
```

The frontend will be available at http://localhost:3000

### Backend Setup

```bash
cd src/api
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The API will be available at http://localhost:8000

### Environment Variables

Copy `.env.example` to `.env` and configure:

```
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5-nano
```

## Development

See `.agent/blueprints/omnirank-web-platform.md` for the full development plan.

## License

MIT
