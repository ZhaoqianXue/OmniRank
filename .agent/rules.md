# Rules

## Tech Stack

### Frontend

| Technology | Purpose |
|------------|---------|
| React 18 | UI Framework |
| Next.js 15 | Full-stack Framework (App Router) |
| TypeScript | Type Safety |
| Three.js | 3D Visualization |
| Tailwind CSS | Styling |
| shadcn/ui | UI Components |
| Framer Motion | Animations |
| Recharts | Data Visualization |
| Lucide Icons | Icon Library |

### Backend

| Technology | Purpose |
|------------|---------|
| FastAPI | REST API Framework |
| LangGraph | Agentic Workflow Orchestration |
| LangChain | LLM Integration |
| Pydantic | Data Validation |
| OpenAI GPT | Large Language Model |

## AI File and Code Generation Standards

### Objective

Standardize the structure and paths of AI-generated content (documents, code, test files, etc.) to avoid polluting the root directory or creating confusing naming conventions.

### Project Structure Conventions

#### Standard Project Directory Structure

Applicable to any medium-to-large software or research engineering project.

##### Top-Level Directory Structure

```
project/
├── .agent/                # AI Agent configuration and memory
│   ├── blueprints/        # Core design, architecture, and proposals
│   ├── skills/            # Custom agent skills and workflows
│   ├── scripts/           # Agent-specific utility scripts
│   └── rules.md           # This file (Project Rules)
├── README.md              # Project description and overview
├── requirements.txt       # Dependencies
├── .gitignore             # Git ignore rules
├── .env                   # Environment variables
├── src/                   # Core source code (Python/TS)
├── tests/                 # Test suites (TDD)
├── docs/                  # Documentation and Literature Review
├── data/                  # Raw and processed datasets
├── scripts/               # Project-level tools and batch tasks
├── results/               # Reports, charts, and scientific outputs
└── docker/                # Containerization deployment related (Dockerfile, compose)

```

##### Source Code Structure (`src/`)

```
src/
├── __init__.py
├── main.py                # Program entry point
├── core/                  # Core logic (algorithms, models, pipelines)
├── modules/               # Functional modules (API, services, tasks)
├── utils/                 # Common utility functions
├── interfaces/            # Interface layer (REST/gRPC/CLI)
├── config/                # Default configuration
├── data/                  # Data access layer (DAO, repository)
└── pipelines/             # Workflow or task scheduling logic
```

##### Test Structure (`tests/`)

```
tests/
├── unit/                  # Unit tests
├── integration/           # Integration tests
├── e2e/                   # End-to-end tests
└── fixtures/              # Test data and mocks
```

##### Experimental Projects Structure (AI/ML)

```
experiments/
├── configs/               # Experiment configurations
├── runs/                  # Results and logs for each run
├── checkpoints/           # Model weights
├── metrics/               # Performance metric records
└── analysis/              # Result analysis scripts
```

##### Versioning and Environment Management

- `venv/` or `.venv/`: Virtual environment (not in repo)
- `Makefile` or `tasks.py`: Standardized task execution (build/test/deploy)
- `.pre-commit-config.yaml`: Code quality hooks
- `.github/workflows/`: CI/CD pipelines

#### Structure Benefits

This structure provides:
- **Clear logical layering**
- **Independent deployment, testing, and documentation**
- **Extensible, collaborative, and versionable**

Can be adapted to specific languages or frameworks (Python/Node/Go/Java, etc.) as needed.

### File Generation Rules

| File Type | Storage Path | Naming Convention | Notes |
|-----------|--------------|-------------------|-------|
| Python Source | `/src` | Module name lowercase, underscore separated | Follow PEP8 |
| Test Code | `/tests` | `test_module_name.py` | Use pytest format |
| Documentation (Markdown) | `/docs` | Use module name plus description, e.g., `module_name_description.md` | UTF-8 encoding |
| Temporary Output or Archives | `/output` | Auto-generate timestamp suffix | Can be auto-cleaned |

### AI Generation Conventions

When AI generates files or code, the following rules **MUST** be followed:

#### Mandatory Rules

1. **DO NOT** create files in the root directory (unless it's a standard config like .env)
2. All new content MUST align with the definitions in `.agent/blueprints/`
3. All new files must be placed in the correct categorized folder
4. File names should be readable and semantic
5. Use English for all code, comments, and formal documentation

#### Default Paths

If file path is not explicitly specified, default to:
- Code → `/src`
- Tests → `/tests`
- Documentation → `/docs`
- Temporary content → `/output`

### Summary

> **CRITICAL**: Follow the project structure:
>
> - Source code goes into `/src`
> - Test code goes into `/tests`
> - Documentation goes into `/docs`
> - **DO NOT create any files in the root directory**
> - Ensure compliance with naming conventions
> - All code must be in English (no Chinese characters)

## Automatic Skill Integration

Instead of manual prompting, the agent should maintain a **"Skill-First" mindset**. For every task, the agent must:

1. **Self-Check**: Proactively scan `.agent/skills/` to see if any installed skill (e.g., planning, academic writing, TDD, or visualization) can enhance the current task's quality or rigor.
2. **Context-Driven Activation**: Automatically load and apply a skill if the task context aligns with the skill's purpose.
3. **Flexible Application**: Use judgment to adapt skills to the specific context, ensuring that skills serve as a professional multiplier rather than a rigid constraint.

## Writing Style

1. **State facts directly** — Do not justify obvious choices
2. **Assume reader competence** — Do not explain common knowledge
3. **Avoid defensive framing** — Do not preemptively address hypothetical criticism

**Avoid:** "A critical design decision is...", "Unlike X, we...", "This ensures/guarantees...", "Addressing concerns...", "It is important to note..."

## Browser and Background Execution

1. **Prioritize Background Execution**: For search queries, information retrieval, and testing, prioritize using background tools (e.g., `search_web`, `read_url_content`, CLI-based testing tools) rather than opening a browser window.
2. **Explicit Browser Request**: Only open a browser window for searching or testing if explicitly requested by the user.
