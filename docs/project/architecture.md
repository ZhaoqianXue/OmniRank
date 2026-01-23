# OmniRank: LLM Agent System Architecture

## Executive Summary

OmniRank is an **agentic framework** that combines the reasoning capabilities of Large Language Models (LLMs) with the mathematical rigor of **spectral ranking inferences**. The system employs a **decoupled architecture** where LLM agents parse user queries and orchestrate the analysis pipeline, while a specialized **Spectral Calculation Engine** (`ranking_cli.R`) executes the mathematical computations.

This architecture is inspired by the LAMBDA framework published in the *Journal of the American Statistical Association - Applications and Case Studies*.

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OmniRank System Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     USER INTERACTION LAYER                           │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │    │
│  │  │  Natural Lang.  │  │   Data Upload   │  │   Visualization     │  │    │
│  │  │    Interface    │  │   (CSV/JSON)    │  │    Dashboard        │  │    │
│  │  └────────┬────────┘  └────────┬────────┘  └──────────▲──────────┘  │    │
│  └───────────┼────────────────────┼───────────────────────┼────────────┘    │
│              │                    │                       │                  │
│              ▼                    ▼                       │                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      LLM AGENT ORCHESTRATION LAYER                   │    │
│  │                                                                      │    │
│  │              ┌──────────────────────────────────────┐               │    │
│  │              │         FIXED PIPELINE               │               │    │
│  │              │   (Triggered by data upload)         │               │    │
│  │              └──────────────────┬───────────────────┘               │    │
│  │                                 │                                    │    │
│  │         ┌───────────────────────┴───────────────────┐               │    │
│  │         ▼                                           ▼               │    │
│  │  ┌────────────┐      ┌────────────┐      ┌────────────────────┐    │    │
│  │  │  DATA      │      │  ENGINE    │      │  ANALYST AGENT     │    │    │
│  │  │  AGENT     │─────▶│ORCHESTRATOR│─────▶│                    │    │    │
│  │  │            │      │            │      │ • Report generation│    │    │
│  │  │ • Schema   │      │ • Invoke   │      │ • Visualization    │    │    │
│  │  │   parsing  │      │   R script │      │ • Q&A with user    │    │    │
│  │  │ • Format   │      │ • Output   │      │ • Error diagnosis  │    │    │
│  │  │   convert  │      │   parse    │      │   (ReAct loop)     │    │    │
│  │  │ • bigbetter│      │            │      │                    │    │    │
│  │  │   inference│      │            │      │                    │    │    │
│  │  └─────┬──────┘      └────────────┘      └──────────┬─────────┘    │    │
│  │        │                                            │               │    │
│  │        └────────── Error Correction ◄───────────────┘               │    │
│  │                                                                      │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   SPECTRAL CALCULATION ENGINE                        │    │
│  │                                                                      │    │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │    │
│  │  │   HYPERGRAPH   │  │   MARKOV CHAIN │  │   UNCERTAINTY          │ │    │
│  │  │  CONSTRUCTOR   │  │   ANALYZER     │  │   QUANTIFICATION       │ │    │
│  │  │                │  │                │  │                        │ │    │
│  │  │ • Multiway     │  │ • Transition   │  │ • Asymptotic variance  │ │    │
│  │  │   comparison   │  │   matrix P     │  │ • Bootstrap CI         │ │    │
│  │  │   encoding     │  │ • Stationary   │  │ • Rank confidence      │ │    │
│  │  │ • Heterogen.   │  │   distribution │  │   intervals            │ │    │
│  │  │   edge sizes   │  │ • Eigenvector  │  │ • Two-sample testing   │ │    │
│  │  │ • Weight func  │  │   computation  │  │                        │ │    │
│  │  │   f(A_l)       │  │                │  │                        │ │    │
│  │  └───────┬────────┘  └───────┬────────┘  └───────────┬────────────┘ │    │
│  │          │                   │                       │              │    │
│  │          ▼                   ▼                       ▼              │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │              SPECTRAL RANKING INFERENCE CORE                  │   │    │
│  │  │                                                               │   │    │
│  │  │  θ̂_i := log π̂_i - (1/n) Σ log π̂_k                           │   │    │
│  │  │                                                               │   │    │
│  │  │  • Minimax optimal estimation                                 │   │    │
│  │  │  • O(n³) complexity via eigen-decomposition                   │   │    │
│  │  │  • Two-step optimal weighting                                 │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      KNOWLEDGE BASE LAYER                            │    │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │    │
│  │  │   DOMAIN       │  │   METHOD       │  │   TEMPLATE             │ │    │
│  │  │   KNOWLEDGE    │  │   LIBRARY      │  │   REPOSITORY           │ │    │
│  │  │                │  │                │  │                        │ │    │
│  │  │ • BTL model    │  │ • Spectral     │  │ • Code templates       │ │    │
│  │  │ • PL model     │  │   methods      │  │ • Prompt templates     │ │    │
│  │  │ • Applications │  │ • Bootstrap    │  │ • Report templates     │ │    │
│  │  │   examples     │  │ • Testing      │  │ • Visualization        │ │    │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Multi-Agent Collaboration Workflow (Fixed Pipeline)

```
                    ┌──────────────────┐
                    │   User Uploads   │
                    │       Data       │
                    └────────┬─────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │          DATA AGENT          │
              │  ┌────────────────────────┐  │
              │  │ 1. Format Recognition  │  │
              │  │ 2. Schema Inference    │  │
              │  │    (BigBetter, Items, Indicator)  │  │
              │  │ 3. Data Validation     │  │
              │  └────────────────────────┘  │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   INTERACTIVE CONFIG (UI)    │
              │  ┌────────────────────────┐  │
              │  │ User verifies schema   │  │
              │  │ & adjusts parameters   │  │
              │  └────────────────────────┘  │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │     ENGINE ORCHESTRATOR      │
              │  ┌────────────────────────┐  │
              │  │ 1. Synthesize Params   │  │
              │  │ 2. Invoke ranking_cli.R│  │
              │  │ 3. Parse JSON Output   │  │
              │  └─────────┬──────────────┘  │
              └────────────┼─────────────────┘
                           │
            ┌──────────────┴──────────────┐
            │         Status?             │
            │  ┌───────┐      ┌────────┐  │
            │  │ Error │      │Success │  │
            │  └───────┘      └────────┘  │
            │      │               │      │
            ▼      ▼               ▼      ▼
    ┌────────────────────┐    ┌────────────────────┐
    │   ANALYST AGENT    │    │   ANALYST AGENT    │
    │ (Error Diagnosis)  │    │ (Report Generation)│
    │                    │    │                    │
    │ • Diagnose Cause   │    │ • Generate Plots   │
    │ • Suggest Fix      │    │ • Write Report     │
    │                    │    │ • Answer User Q&A  │
    └──────────┬─────────┘    └─────────┬──────────┘
               │                        │
               │                        ▼
               │              ┌──────────────────┐
    To Data or │              │   Final Output   │
    Orchestrator              │ • Rankings       │
               │              │ • Visualizations │
               │              │ • Analysis Report│
               │              └──────────────────┘
```

### 1.3 Self-Correcting Mechanism

```
┌──────────────────────────────────────────────────────────────────────┐
│                     SELF-CORRECTING MECHANISM                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐                                                     │
│  │ User Query  │                                                     │
│  │ + Data      │                                                     │
│  └──────┬──────┘                                                     │
│         ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    ENGINE ORCHESTRATOR                       │    │
│  │  Invoke ranking_cli.R with configured parameters             │    │
│  └──────────────────────────┬──────────────────────────────────┘    │
│                             │                                        │
│                             ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    SPECTRAL ENGINE                           │    │
│  │                    Execute Code                              │    │
│  └──────────────────────────┬──────────────────────────────────┘    │
│                             │                                        │
│                     ┌───────┴───────┐                               │
│                     │    Result?    │                               │
│                     └───────┬───────┘                               │
│               ┌─────────────┴─────────────┐                         │
│               │                           │                         │
│         ┌─────▼─────┐              ┌──────▼─────┐                   │
│         │  SUCCESS  │              │   ERROR    │                   │
│         └─────┬─────┘              └──────┬─────┘                   │
│               │                           │                         │
│               │                           ▼                         │
│               │         ┌────────────────────────────────┐          │
│               │         │         ANALYST AGENT          │          │
│               │         │                                │          │
│               │         │  1. Analyze error message      │          │
│               │         │  2. Identify root cause        │          │
│               │         │  3. Suggest corrections        │          │
│               │         └──────────────┬─────────────────┘          │
│               │                        │                            │
│               │                        ▼                            │
│               │         ┌────────────────────────────────┐          │
│               │         │    n < MAX_ATTEMPTS?           │          │
│               │         └──────────────┬─────────────────┘          │
│               │               ┌────────┴────────┐                   │
│               │               │                 │                   │
│               │         ┌─────▼─────┐    ┌──────▼──────┐            │
│               │         │    YES    │    │     NO      │            │
│               │         │           │    │             │            │
│               │         │ Revise &  │    │ Request     │            │
│               │         │ Re-execute│    │ Human Help  │            │
│               │         └─────┬─────┘    └──────┬──────┘            │
│               │               │                 │                   │
│               │               │      ┌──────────▼──────────┐        │
│               │               │      │  HUMAN INTERVENTION │        │
│               │               │      │  • Review code      │        │
│               │               │      │  • Manual fix       │        │
│               │               │      │  • Provide guidance │        │
│               │               │      └──────────┬──────────┘        │
│               │               │                 │                   │
│               │               └────────┬────────┘                   │
│               │                        │                            │
│               └────────────────────────┼───────────────────────────►│
│                                        │                            │
│                                        ▼                            │
│                         ┌────────────────────────────────┐          │
│                         │        FINAL RESPONSE          │          │
│                         │  • Ranking results + Code      │          │
│                         │  • Natural language summary    │          │
│                         └────────────────────────────────┘          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.4 Spectral Ranking Pipeline Detail

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   SPECTRAL RANKING INFERENCE PIPELINE                     │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │                INPUT: Comparison Data                         │     │
│   │                                                               │     │
│   │   D = {(c_l, A_l)}_{l∈D}                                      │     │
│   │                                                               │     │
│   │   where c_l = winner, A_l = choice set (heterogeneous sizes) │     │
│   └───────────────────────────────┬───────────────────────────────┘     │
│                                   │                                      │
│                                   ▼                                      │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │            STEP 1: HYPERGRAPH CONSTRUCTION                    │     │
│   │                                                               │     │
│   │   Build comparison graph G = {A_l | l ∈ D}                   │     │
│   │                                                               │     │
│   │   Compute index sets:                                         │     │
│   │   • W_j = {l ∈ D | j ∈ A_l, c_l = j}  (j wins)              │     │
│   │   • L_i = {l ∈ D | i ∈ A_l, c_l ≠ i}  (i loses)             │     │
│   └───────────────────────────────┬───────────────────────────────┘     │
│                                   │                                      │
│                                   ▼                                      │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │            STEP 2: TRANSITION MATRIX                          │     │
│   │                                                               │     │
│   │                  ┌ (1/d) Σ 1/f(A_l),  if i ≠ j               │     │
│   │   P_ij =        │      l∈W_j∩L_i                              │     │
│   │                  └ 1 - Σ P_ik,        if i = j               │     │
│   │                       k≠i                                     │     │
│   │                                                               │     │
│   │   Weight function f(A_l):                                     │     │
│   │   • Uniform: f(A_l) = 1                                       │     │
│   │   • Size: f(A_l) = |A_l|                                      │     │
│   │   • Optimal: f(A_l) = Σ exp(θ*_u)  (estimated in step 2b)    │     │
│   │                        u∈A_l                                  │     │
│   └───────────────────────────────┬───────────────────────────────┘     │
│                                   │                                      │
│                     ┌─────────────┴─────────────┐                       │
│                     │                           │                       │
│              STEP 2a (Initial)          STEP 2b (Optimal)               │
│   ┌─────────────────▼──────────────┐  ┌────────▼───────────────────┐   │
│   │   Uniform/Size weighting       │  │  Use θ̂^(initial) to       │   │
│   │   f(A_l) = |A_l|              │→ │  estimate optimal weights  │   │
│   │                                │  │  f(A_l) ≈ Σ exp(θ̂_u)     │   │
│   └────────────────────────────────┘  └────────────────────────────┘   │
│                                   │                                      │
│                                   ▼                                      │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │            STEP 3: EIGEN-DECOMPOSITION                        │     │
│   │                                                               │     │
│   │   Solve: π̂^T P = π̂^T   (stationary distribution)             │     │
│   │                                                               │     │
│   │   Method: Left eigenvector of P for eigenvalue 1              │     │
│   │   Complexity: O(n³) via standard decomposition                │     │
│   │              O(n²) via power iteration                        │     │
│   └───────────────────────────────┬───────────────────────────────┘     │
│                                   │                                      │
│                                   ▼                                      │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │            STEP 4: PREFERENCE SCORE ESTIMATION                │     │
│   │                                                               │     │
│   │   θ̂_i = log π̂_i - (1/n) Σ log π̂_k                           │     │
│   │                         k=1                                   │     │
│   │                                                               │     │
│   │   Identification constraint: Σ θ*_i = 0                       │     │
│   │                              i=1                              │     │
│   └───────────────────────────────┬───────────────────────────────┘     │
│                                   │                                      │
│                                   ▼                                      │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │            STEP 5: UNCERTAINTY QUANTIFICATION                 │     │
│   │                                                               │     │
│   │   Asymptotic variance (Theorem 2):                            │     │
│   │                                                               │     │
│   │             1                 1(i∈A_l)                        │     │
│   │   Var = ────────── Σ  ─────────── (Σ exp(θ_u) - exp(θ_i))exp(θ_i) │
│   │         d²τ_i²   l∈D   f²(A_l)   u∈A_l                       │     │
│   │                                                               │     │
│   │   Bootstrap for critical values (Section 3.4):                │     │
│   │   • Gaussian multiplier: G_M = max|Σ J_kl ω_l|/σ̃_km          │     │
│   │   • Monte Carlo: Q_{1-α} = (1-α)-quantile of G_M             │     │
│   └───────────────────────────────┬───────────────────────────────┘     │
│                                   │                                      │
│                                   ▼                                      │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │            OUTPUT: RANKING INFERENCE RESULTS                  │     │
│   │                                                               │     │
│   │   • Estimated scores: θ̂ = (θ̂_1, ..., θ̂_n)                   │     │
│   │   • Rankings: r̂ = argsort(-θ̂)                               │     │
│   │   • Confidence intervals: [R_mL, R_mU] for each item         │     │
│   │   • Hypothesis test results (if requested)                   │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Multi-Agent Collaboration Design

OmniRank employs a streamlined architecture with a fixed pipeline and two specialized agents.

### 2.1 Agent Roles and Responsibilities

| Component | Primary Role | Key Responsibilities |
|-----------|-------------|---------------------|
| **Fixed Pipeline** | Workflow Control | Trigger on data upload, coordinate agent execution sequence |
| **Data Agent** | Data Processing | Schema parsing, format conversion, validation, **preference direction inference (bigbetter)** |
| **Engine Orchestrator** | Engine Invocation | Receive parameters from Data Agent, invoke `ranking_cli.R`, parse JSON/CSV output |
| **Analyst Agent** | Analysis & Output | Report generation, visualization, user Q&A, error diagnosis |

### 2.2 Short-term Memory Architecture

To maintain coherent multi-turn interactions, OmniRank implements a **short-term memory system** that persists within each analysis session:

```python
@dataclass
class SessionMemory:
    """
    Session-scoped memory for maintaining analysis context.
    """
    user_intent_history: List[ParsedIntent]     # All parsed intents in session
    data_state: DataState                        # Current data schema and validation
    execution_trace: List[ExecutionRecord]       # Log of all engine invocations
    result_cache: Dict[str, RankingResult]       # Cached computation results
    
    def get_relevant_context(self, instruction: str) -> Context:
        """Retrieve context relevant to current instruction."""
        pass
    
    def append_trace(self, params: EngineParams, observation: Observation):
        """Log execution for error diagnosis."""
        pass
```

| Memory Component | Purpose | Example Use Case |
|------------------|---------|------------------|
| User Intent History | Enable follow-up queries | "now show confidence intervals" after ranking |
| Data State | Avoid redundant preprocessing | Iterative analysis on same dataset |
| Execution Trace | Support error diagnosis | Inspector reviews failed attempts |
| Result Cache | Enable comparative queries | "compare with previous results" |

### 2.3 Workflow Algorithm

The workflow follows a fixed pipeline triggered by data upload:

```
Algorithm: OmniRank Workflow

Require: Data Agent (Da), Engine Orchestrator (Eo), Analyst Agent (An), Session Memory (M)
Require: user_data (d), max_attempts (T)

// PHASE 1: Data Processing & Configuration
1:  schema ← Da.infer_schema(d)                 ▷ Infer BigBetter, Items, Indicators
2:  params ← Eo.configure(schema, user_input)   ▷ Interactive Verification
3:  M.update_data_state(schema, params)

// PHASE 2: Computation
4:  n ← 0
5:  result ← Eo.execute(params)                 ▷ Execute spectral engine

// PHASE 3: Error Handling (Analyst diagnoses)
5:  while result.status == ERROR and n < T do
6:      n ← n + 1
7:      error_type, suggestions ← An.diagnose(result.error, M.trace)
8:      if error_type == DATA_ERROR then
9:          processed_data, params ← Da.reanalyze(d, suggestions)
10:     else
11:         params ← Eo.revise_params(params, suggestions)
12:     end if
13:     result ← Eo.invoke_ranking_cli(params)
14: end while
15:
16: if result.status == ERROR then
17:     result ← human_intervention(params, M.trace)  ▷ Human-in-the-loop
18: end if

// PHASE 4: Output Generation
19: report ← An.generate_report(result)
20: visualizations ← An.generate_visualizations(result)
21: M.cache_result(result)
22: return {report, visualizations}

// PHASE 5: Ongoing Q&A (triggered by user follow-up)
// An.answer(query, M.context, spectral_knowledge)
```

**Workflow Phases:**

| Phase | Component | Action |
|-------|-----------|--------|
| Data Processing | Data Agent | Infer schema, bigbetter, and standardize format |
| Configuration | Engine Orchestrator | **Interactive User Configuration** & Validation |
| Computation | Engine Orchestrator | Execute ranking_cli.R in isolated process |
| Error Handling | Analyst Agent | Diagnose errors, request corrections from Data Agent or Engine Orchestrator |
| Output Generation | Analyst Agent | Generate report and visualizations |
| Q&A | Analyst Agent | Answer user follow-up questions using memory + spectral knowledge |

### 2.4 Tool Ecosystem

OmniRank integrates multiple tools beyond the core spectral engine to provide comprehensive analysis capabilities:

| Tool | Function | Input | Output |
|------|----------|-------|--------|
| **Spectral Ranking Engine** | Core ranking computation | CSV comparison data | JSON with θ̂, ranks, CIs |
| **Visualization Generator** | Publication-ready plots | Ranking results JSON | PNG/PDF/HTML figures |
| **Report Generator** | Stakeholder communication | Analysis results | Markdown/PDF reports |

```python
class ToolEcosystem:
    """
    Registry of available tools for the Engine Orchestrator.
    """
    tools = {
        "ranking_engine": {
            "path": "src/spectral_engine/ranking_cli.R",
            "description": "Spectral ranking with bootstrap CI",
            "input_format": "CSV",
            "output_format": "JSON"
        },
        "visualization": {
            "path": "src/tools/visualize.py",
            "description": "Generate ranking visualizations",
            "input_format": "JSON",
            "output_format": "PNG/HTML"
        },
        "report_generator": {
            "path": "src/tools/report.py",
            "description": "Generate analysis reports",
            "input_format": "JSON",
            "output_format": "MD/PDF"
        }
    }
```

---

## 3. Core Components Detailed Design

### 3.1 Data Agent

The Data Agent acts as the intelligent interface between user data and the spectral engine, executing three core functions: **Format Recognition & Standardization**, **Semantic Schema Inference**, and **Data Validation**.

#### 3.1.1 Function 1: Format Recognition & Standardization

```python
class DataStandardizer:
    """
    Identifies and standardizes input data formats for the spectral engine.
    """
    
    SUPPORTED_FORMATS = ["pointwise", "pairwise", "multiway"]
    
    def adjust_to_engine_schema(self, file_path: str) -> StandardizedPath:
        """
        Performs lightweight standardization to ensure compatibility with ranking_cli.R.
        
        Actions:
        - Detect format (Pointwise vs Pairwise/Multiway)
        - Ensure CSV/JSON structure meets R script requirements
        - Standardize column delimiters and encoding
        
        Mapping Strategy:
        | Input Format | Detected Characteristics | Engine Input Specification |
        |--------------|--------------------------|----------------------------|
        | **Pointwise** | Unstructured items & scores | Columns: `Item`, `Score` (numerical) |
        | **Pairwise** | Two distinct item cols | Columns: `Item_1`, `Item_2`, `Winner` (or `Score_1` > `Score_2`) |
        | **Multiway** | Rank-ordered columns | Columns: `Rank_1` (Winner), `Rank_2`, ... `Rank_k` |
        """
        pass
```

#### 3.1.2 Function 2: Semantic Schema Inference

```python
@dataclass
class SemanticSchema:
    bigbetter: int                # 1=descending (accuracy), 0=ascending (latency)
    ranking_items: List[str]      # Entities to rank (e.g., "ChatGPT", "Claude")
    indicator_col: Optional[str]  # Selected categorical dimension (e.g., "Task_Type")
    indicator_values: List[str]   # Extracted unique segments (e.g., ["math", "code"])
    inference_confidence: float   # Confidence score of inference

class SchemaInferer:
    """
    Infers semantic metadata to enable precise user control.
    """
    
    def infer_schema(self, dataframe: pd.DataFrame, user_instruction: str) -> SemanticSchema:
        """
        Infers semantic roles of data columns using hierarchical heuristics:
        
        1. Ranking Items Identification:
           - Scan for Categorical columns with high cardinality relative to row count.
           - Check for entity-like naming (e.g., "Model", "System", "Agent", "Player").
           - Returns: List of unique entity names found in these columns.
           
        2. Ranking Indicator Identification:
           - Scan for Categorical columns with low cardinality.
           - CRITICAL: Picks AT MOST ONE column.
           - Returns: The column name (`indicator_col`) and its unique values (`indicator_values`).
           
        3. Preference Direction (bigbetter) Inference:
           - **Priority 1: User Instruction** (Explicit overrides).
           - **Priority 2: Semantic Analysis**:
             - `bigbetter=1` keywords: "accuracy", "score", "reward", "win", "success".
             - `bigbetter=0` keywords: "error", "loss", "latency", "time", "rank".
           - **Priority 3: Distributional Analysis**:
             - Detect unbounded postive distributions (often counts/scores -> 1).
             - Detect 0-1 bounded distributions (probabilities/accuracies -> 1).
        """
        pass
```

#### 3.1.3 Function 3: Data Validation

```python
class DataValidator:
    """
    Performs targeted sanity checks.
    """
    
    def validate(self, data: StandardizedData, n_items: int) -> List[Warning]:
        """
        Checks for critical issues and statistical feasibility.
        
        1. Sparsity Check (Statistical Warning):
           - Formula: Warn if $M < n \log n$ (M=comparisons, n=items).
           - Rationale: Below this threshold, spectral connectivity is theoretically unstable.
           
        2. Format Consistency (Critical Error):
           - Check for mixed data types in Item columns.
           - Check for undefined values (NaN/Null) in critical columns.
           
        3. Connectivity Check:
           - Uses `networkx` to verify if the comparison graph is connected.
           - WARNING: If disconnected, alerts user that rankings are relative to components.
        """
        pass
```
        """
        pass
```

### 3.2 Engine Orchestrator

The Engine Orchestrator is a **deterministic system component** that serves as the bridge between user configuration and the spectral ranking engine. It implements two core functions: **Interactive Configuration Management** and **Robust Engine Execution**.

#### 3.2.1 Function 1: Interactive Configuration Management

```python
@dataclass
class UserControlParams:
    """Parameters exposed for user adjustment in the UI."""
    # Data Schema Controls
    bigbetter: Optional[int] = None       # Overrides inferred direction
    selected_items: Optional[List[str]] = None # Filter to subset of items
    selected_indicator_values: Optional[List[str]] = None # Values within indicator_col to keep
    
    # Statistical Controls (Advanced)
    bootstrap_iterations: int = 2000      # Default B=2000 for robust CIs
    random_seed: int = 42                 # For reproducibility
    
class ConfigManager:
    """
    Synthesizes final execution parameters from Data Schema and User Controls.
    """
    
    def synthesize_params(self, schema: SemanticSchema, user_config: UserControlParams, 
                         file_path: str, output_dir: str) -> EngineParams:
        """
        Merges automatically inferred schema with user overrides.
        
        Logic:
        1. Preference Direction: User Override > Inferred Schema > Default (1)
        2. Item Filtering: Pass `selected_items` list to R script (if any)
        3. Indicator Selection: Pass `selected_indicator` column name (if any)
        4. Statistical Params: Use User Config values directly
        """
        return EngineParams(
            csv_path=file_path,
            bigbetter=user_config.bigbetter if user_config.bigbetter is not None else schema.bigbetter,
            items_filter=user_config.selected_items,
            indicator_col=user_config.selected_indicator,
            B=user_config.bootstrap_iterations,
            seed=user_config.random_seed,
            output_dir=output_dir
        )
```

#### 3.2.2 Function 2: Robust Engine Execution

```python
class RScriptExecutor:
    """
    Manages the lifecycle of the external R process and output parsing.
    """
    
    R_SCRIPT_PATH = "src/spectral_engine/ranking_cli.R"
    
    def execute_and_parse(self, params: EngineParams, timeout_sec: int = 300) -> RankingResult:
        """
        Executes the R script, captures output, and logs the execution trace.
        
        Steps:
        1. **Execution**: Run R script in subprocess with timeout.
           - Capture STDOUT (JSON) and STDERR (Logs).
           
        2. **Trace Logging**: Create `TraceEntry` with timestamp, params, and logs.
           - Store in `SessionMemory.execution_trace` for potential Analyst diagnosis.
           
        3. **Result Parsing**: 
           - If exit_code == 0: Robustly parse JSON from STDOUT, ignoring R startup warnings.
           - If exit_code != 0: Raise `EngineExecutionError` containing STDERR log.
        """
        cmd = self._build_cmd(params)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
            
            # Log trace regardless of success/failure
            SessionMemory.log_trace(params, proc)
            
            if proc.returncode != 0:
                raise EngineExecutionError(proc.stderr)
                
            return self._parse_json_robust(proc.stdout)
            
        except subprocess.TimeoutExpired:
            SessionMemory.log_timeout(params)
            raise EngineExecutionError("Execution timed out")
```

#### 3.2.3 The ranking_cli.R Script

The core spectral ranking computation is implemented in `src/spectral_engine/ranking_cli.R`. This script is a **validated, production-ready implementation** of the spectral method with the following capabilities:

| Feature | Description |
|---------|-------------|
| **Data Processing** | Automatically detects Pointwise (Wide) or Pairwise (Long, `Winner/Loser`) format. |
| **Spectral Estimation** | Computes θ̂ via transition matrix eigen-decomposition. |
| **Direct Conversion** | Transforms Long format to comparison matrix without intermediate wide-matrix. |
| **Variance Estimation** | Calculates asymptotic variance matrix. |
| **Bootstrap CI** | Gaussian multiplier bootstrap for rank confidence intervals. |
| **Output Formats** | JSON (structured) and CSV (tabular) |

**CLI Interface:**
```bash
Rscript src/spectral_engine/ranking_cli.R \
    --csv data/examples/example_data_pointwise.csv \
    --bigbetter 1 \
    --B 2000 \
    --seed 42 \
    --out results/my_analysis
```

**Output JSON Schema:**
```json
{
    "job_id": "my_analysis",
    "params": {
        "bigbetter": true,
        "B": 2000,
        "seed": 42
    },
    "methods": [
        {
            "name": "model_1",
            "theta_hat": 0.234,
            "rank": 2,
            "ci_two_sided": [1, 3],
            "ci_left": 1,
            "ci_uniform_left": 1
        }
    ],
    "metadata": {
        "n_samples": 165,
        "k_methods": 6,
        "runtime_sec": 3.45
    }
}
```

### 3.3 Analyst Agent

The Analyst Agent is the intelligent frontend of the system, responsible for synthesizing results, engaging with users, and handling exceptions. It implements three core functions: **Comprehensive Reporting & Visualization**, **Context-Aware Q&A**, and **ReAct-based Error Diagnosis**.

#### 3.3.1 Function 1: Comprehensive Reporting & Visualization

```python
class AnalysisSynthesizer:
    """
    Transforms raw statistical results into human-readable insights.
    """
    
    def generate_full_report(self, result: EngineResult, data_meta: DataSchema) -> AnalysisReport:
        """
        Coordinates the generation of textual and visual assets.
        
        1. **Visualization Generation**:
           - **Rank Plot**: Comparison of parameter estimates ($\theta$) with CIs.
           - **Heatmap**: Pairwise win-rate matrix to show dominance patterns.
           - **Network Graph**: Topological view of the comparison graph (connectivity).
           - **Score Distribution**: Bootstrap distribution of preferences.
           
        2. **Textual Report Generation**:
           - **Executive Summary**: Key findings (Winner, critical gaps).
           - **Methodology**: Explanation of spectral method & confidence settings.
           - **Actionable Insights**: Generated by LLM analysis of the stats.
        
        Returns: A composite Report object containing Markdown text and Image paths.
        """
        pass
```

#### 3.3.2 Function 2: Context-Aware Q&A

```python
class QAManager:
    """
    Manages interactive dialogue using session context.
    """
    
    def answer_query(self, query: str, session: SessionMemory) -> str:
        """
        Retrieves context and generates answer.
        
        Retrieval Strategy:
        1. **Inspect Result Cache**: Does the query ask about specific rankings/scores in the current result?
           - e.g., "Is Model A better than B?" -> Check CI overlap in `session.result`.
           
        2. **Inspect Data State**: Does it ask about data properties?
           - e.g., "How many math comparisons?" -> Check `session.data_schema`.
           
        3. **Inspect Execution Trace**: Does it ask about the process?
           - e.g., "Did it converge?" -> Check `session.execution_trace`.
           
        Action:
        - Construct prompt with retrieved context + Spectral Theory Knowledge Base.
        - Generate natural language response citing specific evidence.
        """
        pass
```

#### 3.3.3 Function 3: ReAct-based Error Diagnosis

```python
class DiagnosisAgent:
    """
    Implements the ReAct paradigm (Observe-Reason-Act) for self-correction.
    """
    
    ERROR_PATTERNS = {
        "DATA_ERROR": [r"invalid bigbetter", r"format mismatch", r"missing values"],
        "EXECUTION_ERROR": [r"singular matrix", r"numerical overflow", r"timeout"]
    }
    
    def diagnose_and_act(self, error: ExecutionError, trace: TraceEntry) -> CorrectionPlan:
        """
        Executes the ReAct Loop:
        
        1. **OBSERVE**: Read error message (`error.msg`) and execution log (`trace.stderr`).
        
        2. **REASON**: Classify error type.
           - If pattern matches `DATA_ERROR` -> Issues originated upstream.
           - If pattern matches `EXECUTION_ERROR` -> Issues in engine parameters.
           
        3. **ACT**: Formulate a Correction Plan.
           - **Instruction**: Specific guidance for the target agent.
           - **Target**: `DataAgent` (for re-analysis) or `EngineOrchestrator` (for re-config).
           
        Example:
        - Error: "Singular Matrix"
        - Reason: Data too sparse for standard inversion.
        - Act: Instruct EngineOrchestrator to increase regularization or enable approximation.
        """
        pass
```
            )
```

---

## 4. Spectral Calculation Engine

The Spectral Calculation Engine is the **mathematical core** of OmniRank, implementing the spectral ranking inference methodology from Fan et al. (2023).

### 4.1 Mathematical Foundation

The engine implements the following key computations:

#### 4.1.1 Transition Matrix Construction

For comparison data $\{(c_l, A_l)\}_{l \in D}$:

$$
P_{ij} = \begin{cases}
\frac{1}{d} \sum_{l \in W_j \cap L_i} \frac{1}{f(A_l)}, & \text{if } i \neq j \\
1 - \sum_{k: k \neq i} P_{ik}, & \text{if } i = j
\end{cases}
$$

where:
- $W_j = \{l \in D | j \in A_l, c_l = j\}$ (comparisons where $j$ wins)
- $L_i = \{l \in D | i \in A_l, c_l \neq i\}$ (comparisons where $i$ loses)
- $f(A_l)$ is the weighting function

#### 4.1.2 Spectral Estimator

$$
\tilde{\theta}_i := \log \hat{\pi}_i - \frac{1}{n} \sum_{k=1}^n \log \hat{\pi}_k
$$

where $\hat{\pi}$ is the stationary distribution of $P$.

#### 4.1.3 Asymptotic Variance

$$
\text{Var}(J_i^* | G) = \frac{1}{d^2 \tau_i^2} \sum_{l \in D} \frac{\mathbf{1}(i \in A_l)}{f^2(A_l)} \left( \sum_{u \in A_l} e^{\theta^*_u} - e^{\theta^*_i} \right) e^{\theta^*_i}
$$

#### 4.1.4 Optimal Weighting (Two-Step Procedure)

The optimal weighting function that minimizes asymptotic variance:

$$
f(A_l) \propto \sum_{u \in A_l} e^{\theta^*_u}
$$

This is estimated from the initial (equal-weight) spectral estimator.

### 4.2 Engine Implementation

```python
class SpectralRankingEngine:
    """
    Core engine for spectral ranking computations.
    
    Implements the methodology from:
    Fan, J., Lou, Z., Wang, W., Yu, M. (2023). 
    "Spectral Ranking Inferences Based on General Multiway Comparisons"
    """
    
    def __init__(self, config: EngineConfig = None):
        self.config = config or EngineConfig()
    
    def fit(self, comparisons: ComparisonData, 
            weight_scheme: str = 'two_step') -> RankingResult:
        """
        Fits spectral ranking model to comparison data.
        
        Args:
            comparisons: Multiway comparison data
            weight_scheme: 'uniform', 'size', or 'two_step' (optimal)
        
        Returns:
            RankingResult containing estimated scores and uncertainties
        """
        # Step 1: Initial estimation with uniform weights
        theta_initial, pi_initial = self._compute_spectral_estimator(
            comparisons, weight_func=lambda A: 1.0
        )
        
        if weight_scheme == 'two_step':
            # Step 2: Re-estimate with optimal weights
            optimal_weight = lambda A: sum(np.exp(theta_initial[i]) for i in A)
            theta_hat, pi_hat = self._compute_spectral_estimator(
                comparisons, weight_func=optimal_weight
            )
        else:
            theta_hat, pi_hat = theta_initial, pi_initial
        
        # Compute asymptotic variance
        variance = self._estimate_variance(theta_hat, comparisons)
        
        return RankingResult(
            theta_hat=theta_hat,
            pi_hat=pi_hat,
            variance=variance,
            ranks=np.argsort(-theta_hat) + 1
        )
    
    def confidence_intervals(self, result: RankingResult, 
                             comparisons: ComparisonData,
                             alpha: float = 0.05,
                             interval_type: str = 'two_sided') -> np.ndarray:
        """
        Constructs confidence intervals for ranks.
        
        Uses Gaussian multiplier bootstrap following Theorem 5.
        """
        pass
    
    def hypothesis_test(self, result1: RankingResult, result2: RankingResult,
                        test_type: str, **kwargs) -> TestResult:
        """
        Performs hypothesis testing on rankings.
        
        Supported tests:
        - 'top_k_placement': Test if item m is in top-K
        - 'rank_preservation': Test if rank of item m is preserved (two-sample)
        - 'top_k_set': Test if top-K sets are equal (two-sample)
        """
        pass
    
    def _compute_spectral_estimator(self, comparisons: ComparisonData,
                                     weight_func: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """
        Core spectral estimation computation.
        """
        n = comparisons.n_items
        
        # Build transition matrix
        P = np.zeros((n, n))
        
        for winner, choice_set in comparisons.comparisons:
            weight = weight_func(choice_set)
            for loser in choice_set:
                if loser != winner:
                    P[loser, winner] += 1.0 / weight
        
        # Normalize
        d = self._compute_normalization_constant(P, n)
        P = P / d
        np.fill_diagonal(P, 1 - np.sum(P, axis=1) + np.diag(P))
        
        # Eigen-decomposition for stationary distribution
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        idx = np.argmax(np.abs(eigenvalues))
        pi_hat = np.real(eigenvectors[:, idx])
        pi_hat = np.abs(pi_hat) / np.sum(np.abs(pi_hat))
        
        # Transform to log-scale preference scores
        log_pi = np.log(np.maximum(pi_hat, 1e-10))
        theta_hat = log_pi - np.mean(log_pi)
        
        return theta_hat, pi_hat
    
    def _estimate_variance(self, theta_hat: np.ndarray, 
                           comparisons: ComparisonData) -> np.ndarray:
        """
        Estimates asymptotic variance following Equation (6).
        """
        pass
```

---

## 5. User Interface Components

### 5.1 Natural Language Interface

```python
class NaturalLanguageInterface:
    """
    Processes natural language instructions for ranking analysis.
    
    Example queries:
    - "Rank these 10 products based on the pairwise preference data I've uploaded"
    - "Give me 95% confidence intervals for the rankings"
    - "Test whether product A is in the top 3"
    - "Compare the rankings from Q1 and Q2 surveys - have preferences changed?"
    """
    
    def parse_instruction(self, instruction: str, 
                          data_context: dict) -> ParsedInstruction:
        """
        Parses natural language instruction into structured task specification.
        """
        system_prompt = """
        You are an expert in statistical ranking analysis. Parse the user's 
        instruction into a structured task specification.
        
        Available task types:
        - RANKING_ESTIMATION: Estimate preference scores
        - CONFIDENCE_INTERVAL: Construct rank confidence intervals
        - HYPOTHESIS_TESTING: Test ranking hypotheses
        - TOP_K_IDENTIFICATION: Identify top-K items
        - TWO_SAMPLE_COMPARISON: Compare rankings across samples
        """
        
        return self.llm.parse(system_prompt, instruction, data_context)
```

### 5.2 Visualization Dashboard

The visualization component provides interactive exploration of ranking results:

- **Ranking Plot**: Items ordered by estimated preference scores with confidence intervals
- **Transition Matrix Heatmap**: Visualization of the Markov chain structure
- **Uncertainty Quantification**: Rank confidence interval display
- **Comparison Graph**: Network visualization of the comparison hypergraph

---

## 6. System Workflow Examples

### 6.1 Example: Sports Team Ranking

```
User Input:
"I have match results from a soccer tournament. Each match has 2-4 teams 
competing, and I want to rank all 20 teams with statistical confidence."

[Data uploaded: tournament_results.csv]

OmniRank Workflow:

1. PLANNER AGENT:
   - Classifies intent: RANKING_ESTIMATION + CONFIDENCE_INTERVAL
   - Plans workflow with multiway comparison handling

2. DATA AGENT:
   - Parses CSV, identifies multiway comparison format
   - Validates: 156 matches, 20 teams, edge sizes 2-4
   - Constructs comparison hypergraph

3. ENGINE ORCHESTRATOR AGENT:
   - Configures ranking_cli.R parameters
   - Applies two-step optimal weighting
   - Invokes bootstrap CI computation

4. SPECTRAL ENGINE:
   - Executes spectral ranking computation
   - Computes 95% confidence intervals

5. INFERENCE AGENT:
   Produces interpretation:
   
   "Ranking Analysis Results (20 teams, 156 matches):
   
   **Top 5 Teams:**
   1. Team A (θ̂ = 1.82) - Rank CI: [1, 2]
   2. Team B (θ̂ = 1.56) - Rank CI: [1, 3]
   3. Team C (θ̂ = 1.34) - Rank CI: [2, 4]
   4. Team D (θ̂ = 1.21) - Rank CI: [3, 5]
   5. Team E (θ̂ = 0.98) - Rank CI: [4, 7]
   
   The narrow confidence intervals for top teams indicate 
   strong statistical evidence for their relative positions.
   Teams A and B have overlapping CIs, suggesting they may
   be statistically indistinguishable at the 95% level."
```

### 6.2 Example: Two-Sample Comparison

```
User Input:
"Compare customer product preferences between 2023 Q1 and 2023 Q4. 
Have the top 5 products changed significantly?"

OmniRank Workflow:

1. Analysis of Q1 data → RankingResult_Q1
2. Analysis of Q4 data → RankingResult_Q4
3. Two-sample Top-K test (K=5, α=0.05)
4. Result interpretation:

   "Two-Sample Top-5 Comparison (Q1 2023 vs Q4 2023):
   
   Hypothesis: H₀: Top-5 products unchanged vs H₁: Top-5 changed
   
   Result: REJECT H₀ at α = 0.05
   
   Key changes:
   - Product X entered top-5 (Q1 rank: 8, Q4 rank: 3)
   - Product Y dropped from top-5 (Q1 rank: 4, Q4 rank: 11)
   
   The 95% confidence sets for top-5 in each period:
   - Q1: {A, B, C, D, E, F, G} (7 candidates)
   - Q4: {A, B, C, X, D, E} (6 candidates)
   
   Since intersection has only 5 elements, we conclude 
   significant preference shift at the 5% significance level."
```

---

## 8. Technical Specifications

### 8.1 LLM Requirements

| Component | Recommended Model | Minimum Capability |
|-----------|------------------|-------------------|
| Planner Agent | GPT-4 / Claude 3.5 | Chain-of-thought reasoning |
| Code Generator | GPT-4 / Codex | Code generation + debugging |
| Inference Agent | GPT-4 / Claude 3.5 | Natural language explanation |
| Inspector Agent | Any LLM | Pattern matching + reasoning |

### 8.2 Computational Requirements

| Task | Complexity | Typical Runtime |
|------|-----------|----------------|
| Spectral Estimation | O(n³) | < 1s for n < 1000 |
| Bootstrap CI (B=1000) | O(B × n²) | < 30s for n < 500 |
| Two-sample Testing | O(n³) | < 2s for n < 1000 |

### 8.3 System Dependencies

```
Python >= 3.9
NumPy >= 1.21
SciPy >= 1.7
scikit-learn >= 1.0
sentence-transformers >= 2.0
openai >= 1.0 (for LLM integration)
flask >= 2.0 (web interface)
plotly >= 5.0 (visualization)
```

---

## 9. Comparison with Related Systems

| Feature | OmniRank | LAMBDA | GPT-4 ADA | ChemCrow |
|---------|----------|--------|-----------|----------|
| **Domain Focus** | Ranking Inference | General Data Analysis | General | Chemistry |
| **Statistical Rigor** | ✓ Minimax optimal | ✓ Standard ML | ✗ Prone to errors | ✓ Domain-specific |
| **Uncertainty Quantification** | ✓ Asymptotic + Bootstrap | ✗ Limited | ✗ None | ✗ Limited |
| **Multi-Agent Architecture** | ✓ 3 specialized components | ✓ 2 agents | ✗ Single | ✓ Multiple tools |
| **Self-Correction** | ✓ Inspector agent | ✓ Inspector agent | ✗ No | ✓ Iterative |
| **Human-in-the-Loop** | ✓ Full support | ✓ Full support | ✗ No | ✓ Limited |
| **Tool Ecosystem** | ✓ Modular | ✓ KV base | ✗ No | ✓ Tool-based |
| **Code Export** | ✓ Python/R | ✓ IPython | ✓ Python | ✗ No |

---

## 10. Conclusion

The OmniRank LLM Agent System Architecture provides a **rigorous, accessible, and extensible** framework for spectral ranking inferences. Key architectural decisions include:

1. **Decoupled Architecture**: Clear separation between LLM reasoning and mathematical computation ensures both accessibility and statistical rigor.

2. **Modular Collaboration**: Specialized components handle distinct aspects of the analysis pipeline (Data, Engine, Analyst).

3. **Self-Correction Mechanism**: The Inspector Agent provides robust error handling with human-in-the-loop fallback.

4. **Tool Ecosystem**: Modular integration of ranking engine, visualization generator, and report generator enables comprehensive analysis workflows.

5. **Comprehensive Uncertainty Quantification**: Full implementation of asymptotic variance estimation, bootstrap confidence intervals, and hypothesis testing.

This architecture successfully addresses the potential reviewer concern about "LLM Agent as a Wrapper" by demonstrating substantive agent participation in:
- Complex intent parsing and workflow planning
- Engine orchestration with parameter configuration
- Error diagnosis and self-correction
- Statistical result interpretation and communication

The design aligns with the standards of top-tier statistical journals while making advanced spectral ranking methods accessible to non-statisticians through natural language interaction.

---

## References

1. Fan, J., Lou, Z., Wang, W., Yu, M. (2023). Spectral Ranking Inferences Based on General Multiway Comparisons. *arXiv preprint arXiv:2308.02918*.

2. Sun, M., Han, R., Jiang, B., Qi, H., Sun, D., Yuan, Y., Huang, J. (2024). LAMBDA: A Large Model Based Data Agent. *Journal of the American Statistical Association - Applications and Case Studies*.

3. Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR*.

4. Schick, T., et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. *NeurIPS*.
