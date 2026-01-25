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
    └── rules/             # Project rules
        └── rules.md
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

## Usage

1. **Start the backend**:
   ```bash
   cd src/api
   source .venv/bin/activate
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. **Start the frontend**:
   ```bash
   cd src/web
   npm run dev
   ```

3. **Open http://localhost:3000** in your browser

4. **Upload a CSV file** with comparison data (pointwise or pairwise format). A data preview loads first, then the Data Agent infers the schema.

5. **Configure analysis parameters** and click "Start Analysis"

6. **View results** in the Rankings, Heatmap, or Network tabs

## Example Data

Example datasets are provided in `data/examples/`:
- `example_data_pointwise.csv` - Pointwise comparison format
- `example_data_pairwise.csv` - Pairwise comparison format

## Development

See `.agent/blueprints/omnirank-web-platform.md` for the full development plan.

## Project Status

All development phases complete:
- Phase 1: Foundation Setup ✅
- Phase 2: Core Backend - Agent System ✅
- Phase 3: API Layer ✅
- Phase 4: Frontend - Core UI ✅
- Phase 5: Visualizations ✅
- Phase 6: Integration and Polish ✅

## License

MIT
