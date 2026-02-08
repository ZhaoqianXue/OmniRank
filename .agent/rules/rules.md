---
trigger: always_on
---

# Rules

## Monorepo Development Strategy

This project follows a **Monorepo** structure. This strategy ensures AI assistants can maintain full-stack context while keeping concerns separated.

### Rationale

-   **Deep Context Sharing**: Explicitly allows AI to reason across the entire stack (e.g., matching frontend state to backend schemas).
-   **Contract-First Development**: Shared definitions prevent "drift" between producers and consumers.
-   **Atomic Operations**: Features spanning client, server, and core logic are implemented and committed in a single step.
-   **Simplified Orchestration**: Unified scripts and environment management across all Roles.

### Core Structure (Template)

The codebase is organized by **Role** rather than just technology:

```text
src/
├── <client>/      # User Interface & Interaction (Current: web/)
├── <server>/      # API Orchestration & Business Logic (Current: api/)
└── <core>/        # Domain-specific algorithms or legacy logic (Current: spectral_ranking/)
shared/
└── <contracts>/   # Shared types, protocols, and constants (Current: types/)
```

### Universal Guidelines

1.  **Contract-First**: Define API contracts/types in `shared/<contracts>/` *before* implementation.
2.  **Environment Sync**: Root `.env` contains shared configuration; role-specific overrides live in their respective directories.
3.  **Unified Entrypoints**: Use root-level scripts (e.g., `package.json`, `Makefile`) to coordinate cross-role tasks.
4.  **Local Isolation**: Each Role maintains its own dependency manifest (e.g., `requirements.txt`, `package.json`) to allow independent deployment and testing.
5.  **Context Discovery**: Upon first interaction, the Agent must verify the actual directory names mapping to these Roles.

## Tech Stack

### Frontend (<client>)

| Technology | Purpose |
| :--- | :--- |
| React 18 | UI Framework |
| Next.js 15 | Full-stack Framework (App Router) |
| TypeScript | Type Safety |
| Three.js | 3D Visualization |
| Tailwind CSS | Styling |
| shadcn/ui | UI Components |
| Framer Motion | Animations |
| Recharts | Data Visualization |
| Lucide Icons | Icon Library |

### Backend (<server>)

| Technology | Purpose |
| :--- | :--- |
| FastAPI | REST API Framework |
| LangGraph | Agentic Orchestration |
| Pydantic | Data Validation |
| OpenAI GPT | Large Language Model |

### Default LLM Configuration

All LLM-powered agents in this project MUST use `gpt-5-mini` as the default model unless explicitly overridden by user configuration.

| Setting | Value |
| :--- | :--- |
| Default Model | `gpt-5-mini` |
| Environment Variable | `OPENAI_MODEL` |
| Fallback | None (fail if model unavailable) |

### LLM Testing Strategy

- During development and testing, the system SHOULD use a real OpenAI API key stored in the root `.env` file (for example `OPENAI_API_KEY=...`) so that LLM-related behavior is validated against the live API.
- Automated or manual tests that exercise LLM calls are expected to run as **online tests** using this `.env` configuration, rather than offline mocks or stubs, unless a test explicitly documents an alternative strategy.
- Application code and agents MUST read credentials exclusively from environment variables (never hard-code keys in the repository), and `.env` MUST NOT be committed to version control.

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
├── .gitignore             # Git ignore rules
├── .env                   # Environment variables (shared across roles)
├── src/                   # Source code organized by Role (see Monorepo Strategy)
├── shared/                # Cross-role contracts, types, and constants
├── tests/                 # Test suites (TDD)
├── docs/                  # Documentation and Literature Review
├── data/                  # Raw and processed datasets
├── scripts/               # Project-level tools and batch tasks
├── results/               # Reports, charts, and scientific outputs
├── experiments/           # AI/ML experiments, runs, and analysis
├── archived/              # Legacy code and unused files
└── docker/                # Containerization deployment related (Dockerfile, compose)

```

##### Source Code Structure (`src/`) - Monorepo by Role

Aligns with the **Monorepo Development Strategy**. Each Role maintains its own structure:

```
src/
├── <client>/              # User Interface (e.g., web/)
│   ├── app/               # Next.js App Router pages
│   ├── components/        # UI components
│   ├── lib/               # Client utilities and API client
│   └── package.json       # Client dependencies
├── <server>/              # API & Business Logic (e.g., api/)
│   ├── main.py            # Server entry point
│   ├── core/              # Core logic (schemas, executors)
│   ├── agents/            # Agent implementations
│   ├── api/               # Route handlers
│   └── requirements.txt   # Server dependencies
└── <core>/                # Domain-specific algorithms (e.g., spectral_ranking/)
    └── *.R                # R scripts or other domain code
```

##### Shared Contracts Structure (`shared/`)

```
shared/
└── <contracts>/           # Cross-role type definitions (e.g., types/)
    ├── api.ts             # API request/response types
    └── index.ts           # Public exports
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
| Python Backend | `/src/<server>/` | Module name lowercase, underscore separated | Follow PEP8 |
| TypeScript Frontend | `/src/<client>/` | camelCase for files, PascalCase for components | Follow ESLint rules |
| Shared Types | `/shared/<contracts>/` | Lowercase with `.ts` extension | Export via index.ts |
| Test Code | `/tests` | `test_module_name.py` or `*.test.ts` | Use pytest/vitest format |
| Documentation (Markdown) | `/docs` | Use module name plus description, e.g., `module_name_description.md` | UTF-8 encoding |
| Temporary Output or Archives | `/output` | Auto-generate timestamp suffix | Can be auto-cleaned |

### AI Generation Conventions

When AI generates files or code, the following rules **MUST** be followed:

#### Mandatory Rules

1.  **Zero Root Pollution**: DO NOT create files in the root directory (except standard configs like `.env`, `README`, or system-level dotfiles).
2.  **Architecture Alignment**: All new content MUST align with the project's core documentation (e.g., `docs/project/` or `docs/architecture/`).
3.  **Correct Categorization**: All new files must be placed in the predefined folder structures defined in the "Project Structure Conventions" section.
4.  **Semantic Naming**: File names should be descriptive, lowercase, and use underscores/hyphens consistently with the existing codebase.
5.  **Language**: Use English for all code, comments, and formal documentation.
6.  **Style Constraints**: **NO EMOJIS** are allowed in any Markdown (`.md`) documents.

#### Default Paths

If file path is not explicitly specified, default to:
- Frontend code → `/src/<client>/` (e.g., `/src/web/`)
- Backend code → `/src/<server>/` (e.g., `/src/api/`)
- Domain algorithms → `/src/<core>/` (e.g., `/src/spectral_ranking/`)
- Shared types/contracts → `/shared/<contracts>/` (e.g., `/shared/types/`)
- Tests → `/tests`
- Documentation → `/docs`
- Experiments → `/experiments`
- Temporary content → `/output`
- Archived/Legacy content → `/archived`

### Summary

> **CRITICAL**: Follow the Monorepo structure:
>
> - Frontend code goes into `/src/<client>/` (e.g., `/src/web/`)
> - Backend code goes into `/src/<server>/` (e.g., `/src/api/`)
> - Domain algorithms go into `/src/<core>/` (e.g., `/src/spectral_ranking/`)
> - Shared contracts go into `/shared/<contracts>/` (e.g., `/shared/types/`)
> - Test code goes into `/tests`
> - Documentation goes into `/docs`
> - **DO NOT create any files in the root directory** (except standard configs)
> - Ensure compliance with naming conventions
> - All code must be in English (no Chinese characters)
> - **NO EMOJIS** are allowed in any Markdown (.md) documents

## Read-Only Reference Directory

**IMPORTANT**: The `Spectral_Ranking` folder is **READ-ONLY** and serves **ONLY** as references.

### Rules:
- **DO NOT** modify any files within the `Spectral_Ranking` folder
- **DO NOT** use any code from this folder directly in the project
- **ONLY** read and reference files in this folder to understand spectral ranking implementations, simulations, and real-world data applications.
- All files in this folder are for reference purposes only

### Purpose:
- **Spectral_Ranking**: Contains reference implementations, simulations, and real-data applications of spectral ranking methods.

Use this solely as a learning resource, not as a codebase to modify or extend.

## Automatic Skill Integration

Instead of manual prompting, the agent should maintain a **"Skill-First" mindset**. For every task, the agent must:

1. **Self-Check**: Proactively scan `.agent/skills/` to see if any installed skill (e.g., planning, academic writing, TDD, or visualization) can enhance the current task's quality or rigor.
2. **Context-Driven Activation**: Automatically load and apply a skill if the task context aligns with the skill's purpose.
3. **Flexible Application**: Use judgment to adapt skills to the specific context, ensuring that skills serve as a professional multiplier rather than a rigid constraint.

## Writing Style

1. **State facts directly** — Do not justify obvious choices
2. **Assume reader competence** — Do not explain common knowledge
3. **Avoid defensive framing** — Do not preemptively address hypothetical criticism

## Browser and Background Execution

1. **Prioritize Background Execution**: For search queries, information retrieval, and testing, prioritize using background tools (e.g., `search_web`, `read_url_content`, CLI-based testing tools) rather than opening a browser window.
2. **Explicit Browser Request**: Only open a browser window for searching or testing if explicitly requested by the user.