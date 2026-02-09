# Standard Operating Procedure

## Title
OmniRank: A Large-Language-Model Agent Platform for Statistically Rigorous Ranking Inference from Arbitrary Multiway Comparisons

## Target Journal
Journal of the American Statistical Association - Applications and Case Studies

## Background
- Mengxin Yu, the author of `.agent/literature/spectral_ranking_inferences.md`, proposed a ranking inference method based on spectral theory that can handle arbitrary multiway comparison data with statistical optimality. Currently, this method primarily exists as an R package, requiring users to have a specific statistical background and programming skills.
- Mengxin Yu expects me to draw inspiration from several published top-tier journal articles regarding LLM agent platforms (their publication validates the feasibility of their LLM agent architectures). We aim to build an LLM-based agent platform that encapsulates the spectral ranking method into a user-friendly tool, enabling users without a statistical background to conveniently perform ranking inference.
- Reviewers from the target journal may question the architecture of OmniRank. Therefore, we need to reference published LLM agent research in top journals to ensure that the LLM agent's role in OmniRank is substantial enough to avoid being dismissed by reviewers as a simple API wrapper (LLM Agent as a Wrapper).
- OmniRank will adopt a top-down construction approach: first completing the manuscript to a level comparable with top-tier LLM agent publications and meeting the standards of premier journals, followed by the implementation of the OmniRank codebase.
- Already Published Articles
    1. `.agent/literature/automated_hypothesis_validation.md`
    2. `.agent/literature/clinical_prediction_models.md`
    3. `.agent/literature/lambda.md`
    4. `.agent/literature/tissuelab.md`
    Specifically, `.agent/literature/lambda.md` should be used as a primary reference for writing style and structure (without copying specific content), as its publication in the *Journal of the American Statistical Association - Applications and Case Studies* indicates that its writing quality meets the requirements of top-tier journals.

## LLM Agentic Engineering Knowledge Base

**To ensure the autonomy and reliability of the single llm agent system, this project must strictly adhere to the engineering standards detailed in the following documentation, each of which MUST be read in its entirety:**

- [Anthropic: Long-Running Agents](../knowledge/context_engineering/anthropic_long_running_agents.md)
- [Anthropic: Effective Context Engineering](../knowledge/context_engineering/anthropic_context_engineering.md)
- [Manus: Context Engineering](../knowledge/context_engineering/manus_context_engineering.md)

## Contributions

This paper makes the following contributions:

1. **Accessible Spectral Ranking**: We present OmniRank, the first natural language interface for spectral ranking inference, democratizing access to minimax-optimal ranking methods for practitioners without statistical programming expertise. Unlike standard LLMs that are prone to hallucinations in arithmetic tasks, OmniRank decouples instruction following from computation via a specialized Spectral Calculation Engine.

2. **Semantic Schema Inference**: We develop an LLM-based Data Agent capable of automatically inferring comparison data semantics (preference direction, ranking items, indicators), reducing user configuration burden while maintaining statistical rigor. Our evaluation demonstrates xyz% format detection accuracy and xyz% semantic schema inference accuracy across xyz test datasets.

3. **Integrated Uncertainty Quantification**: OmniRank provides automatic generation of bootstrap confidence intervals and rank inference reports based on Fan et al.'s (2023) Gaussian multiplier bootstrap method, enabling practitioners to make statistically grounded decisions without manual bootstrap implementation.

4. **Empirical Validation**: We demonstrate through synthetic and real-world case studies that OmniRank produces results statistically equivalent to manual R implementation, while significantly reducing time-to-insight for non-programmer users.

### Contribution Positioning Strategy

Following LAMBDA's successful positioning in JASA-ACS, OmniRank explicitly positions itself as a **bridge** rather than a statistical innovation:

| What OmniRank IS | What OmniRank IS NOT |
|------------------|----------------------|
| A bridge that democratizes spectral ranking | A new statistical method |
| An intelligent interface to existing theory | A replacement for rigorous methodology |
| A tool to lower the "coding barrier" | A black-box that obscures computation |
| A platform for reproducible ranking analysis | A one-off analysis script |

**Key Messaging**: OmniRank enables domain experts (biologists, sociologists, computer scientists, etc.) to leverage the minimax-optimal spectral ranking methods without requiring linear algebra expertise or R programming skills. The statistical foundations remain unchanged; only the accessibility improves.

## OmniRank Agent Architecture

### Single Agent + Tool Calling

OmniRank employs a Single Agent architecture with Tool Calling (powered by **gpt-5-mini**). The agent operates within a single context window and invokes specialized tools for data processing, spectral computation, and analysis.

### High-Level Architecture

```
                          OmniRank Single Agent
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │                    OmniRank Agent (gpt-5-mini)                  │
    │                    ───────────────────────────                  │
    │                    Single context window                         │
    │                    Tool Calling enabled                          │
    │                                                                  │
    ├──────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │   ┌─────────────────────────────────────────────────────────┐   │
    │   │                    TOOL REGISTRY (10 Tools)              │   │
    │   ├─────────────────────────────────────────────────────────┤   │
    │   │                                                          │   │
    │   │  ┌─────────────────────┐   ┌─────────────────────────┐  │   │
    │   │  │   DATA TOOLS (5)    │   │   ANALYSIS TOOLS (3)    │  │   │
    │   │  │   ─────────────     │   │   ───────────────       │  │   │
    │   │  │   read_data_file    │   │   generate_report       │  │   │
    │   │  │   infer_semantic    │   │   generate_visualizations│  │   │
    │   │  │   _schema           │   │   answer_question       │  │   │
    │   │  │   validate_data     │   └─────────────────────────┘  │   │
    │   │  │   _format           │                                 │   │
    │   │  │   validate_data     │                                 │   │
    │   │  │   _quality          │                                 │   │
    │   │  │   preprocess_data   │                                 │   │
    │   │  └─────────────────────┘                                 │   │
    │   │                                                          │   │
    │   │  ┌─────────────────────┐   ┌─────────────────────────┐  │   │
    │   │  │   ENGINE TOOL (1)   │   │ USER INTERACTION (1)    │  │   │
    │   │  │   ───────────       │   │ ─────────────────       │  │   │
    │   │  │   execute_spectral  │   │ request_user            │  │   │
    │   │  │   _ranking          │   │ _confirmation           │  │   │
    │   │  └─────────────────────┘   └─────────────────────────┘  │   │
    │   │                                                          │   │
    │   └─────────────────────────────────────────────────────────┘   │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
```

### Tool Calling Workflow

The Agent enforces the OmniRank Fixed Pipeline through sequential tool calling.
```
Phase 1: Data Processing
────────────────────────
    read_data_file(file_path)
           │
           ├──[Error]──► Return error to user
           ▼
    infer_semantic_schema(data_summary, user_hints=None)
           │
           ├──[Ambiguous inference]──► Flag for user review
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    FORMAT VALIDATION LOOP                        │
    │  ┌────────────────────────────────────────────────────────────┐ │
    │  │                                                             │ │
    │  │  validate_data_format(current_file_path, schema)            │ │
    │  │         │                                                   │ │
    │  │         ├──[PASS]──────────────────────────► EXIT LOOP ─────┼─┼──► (proceed to quality)
    │  │         │                                                   │ │
    │  │         ├──[FIXABLE]──► preprocess_data(file, schema)       │ │
    │  │         │                      │                            │ │
    │  │         │                      ├─► Update current_file_path │ │
    │  │         │                      └─► (loop back to validate)◄─┘ │
    │  │         │                                                     │
    │  │         └──[UNFIXABLE]──► Return error to user                │
    │  │                                                               │
    │  └───────────────────────────────────────────────────────────────┘
    └──────────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    QUALITY VALIDATION (once)                     │
    │                                                                  │
    │  validate_data_quality(current_file_path, schema)                │
    │         │                                                        │
    │         ├──[PASS]──► Proceed to user confirmation                │
    │         │                                                        │
    │         ├──[Warnings only]──► Proceed with warnings attached     │
    │         │                                                        │
    │         └──[Critical errors]──► Return error to user             │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
           │
           ▼
    request_user_confirmation(proposed_schema, format_result, quality_result)
           │
           ├──[User modifies schema]──► Loop back to infer_schema(user_hints=...)
           │
           ├──[User rejects entirely]──► Ask clarifying questions, re-infer
           ▼
    (Data is ready for Phase 2)

Phase 2: Computation
────────────────────────
    execute_spectral_ranking(config: EngineConfig)
           │
           ├──[R script error]──► Parse error, suggest fixes
           ├──[Convergence warning]──► Include in results
           ▼

Phase 3: Output Generation
────────────────────────
    generate_report(results, session)
           │
           ▼
    generate_visualizations(results, viz_types)
           │
           ▼
    answer_question(question, session)   [User Q&A Loop - multiple iterations]

Q&A Availability (cross-phase):
- Users can ask OmniRank Agent questions at **any time**, not only after report generation.
- Before ranking is completed, `answer_question` must answer with stage-aware guidance and method context.
- After ranking is completed, `answer_question` must use result-level evidence (theta_hat, integer CIs, citation blocks).
- Composer-level `Suggest Question` prompts must be generated from the **user's perspective** (likely next user question), not assistant instructions.
- For latency-sensitive UX, `Suggest Question` uses local deterministic templates (hardcoded, context-aware) rather than round-trip LLM calls.

Suggest Question Product Logic (PM requirements):
- Objective: maximize next-step clarity and reduce user cognitive load in chat.
- Constraint: always return exactly **2** candidate questions.
- UX principle: candidate questions must be actionable, concise, and decision-oriented (avoid meta prompts like "do you want me to...").

Scenario routing (priority order):
1. **Quoted context present**:
   - Priority question should interpret the quoted claim for decision-making.
   - Secondary question should focus on uncertainty/risk (CI caveat or evidence consistency).
   - If multiple quotes exist, one question should ask to reconcile conflicts across quotes.
2. **Session status = error**:
   - Question 1: root cause + fastest fix.
   - Question 2: what can still be trusted/asked while blocked.
3. **Session status = analyzing**:
   - Question 1: what to inspect first once results arrive.
   - Question 2: how CI overlap will affect decision threshold.
4. **Post-schema (configuring)**:
   - Questions should de-risk run configuration (direction, indicator usage, validation checks).
   - If `indicator_col` exists, one question must address segmented vs overall ranking strategy.
5. **Post-analysis (completed)**:
   - Question 1 should be top-vs-runner-up robustness under integer CI interpretation.
   - Question 2 should identify practical tie/tier decisions.
6. **Pre-upload / idle**:
   - Questions should focus on data readiness and method assumptions before running.

Draft-aware refinement:
- If user has partially typed input, use it as the first candidate seed.
- Convert draft into a high-value analytical question (comparison / uncertainty / method / setup / error-fix intent).
- Never output the raw unfinished fragment unchanged unless it is already a clear question.

Quality guardrails:
- Use only user-perspective phrasing.
- Avoid duplicate semantics across the two suggestions.
- Prefer concrete nouns from current context (top item names, indicator name, quote excerpt) over generic wording.
```

## OmniRank Agent Tools Design

### Tool Design Principles

Following Context Engineering best practices from Anthropic and Manus:

**Core Principles:**
1. **Atomicity**: Each tool performs one well-defined operation
2. **Self-Contained**: Tools include all necessary context in their output
3. **JIT Context Loading**: `read_data_file` returns summaries, not full data (file system as externalized memory)
4. **Robust Error Handling**: Each tool returns structured error information that is preserved in context for model learning
5. **Token Efficiency**: Output is minimal but sufficient for LLM reasoning
6. **Clear Parameters**: Input parameters are descriptive, unambiguous, and play to model strengths
7. **Unambiguous Selection**: Each tool has a clearly differentiated purpose; human can definitively say which tool applies
8. **Stable Registry**: Tool definitions remain constant throughout execution; no dynamic tool loading
9. **Append-Only Results**: Tool outputs are append-only; previous actions/observations are never modified

### Tools Overview

#### Data Tools (5)

- **`read_data_file`**
  - Purpose: Read file and return lightweight summary (JIT Context)
  - Input: `file_path: str`
  - Output: `{success: bool, data: Optional[DataSummary], error: Optional[str]}`
  - Note: `DataSummary = {columns: List[str], sample_rows: List[dict], row_count: int, column_types: dict}`

- **`infer_semantic_schema`**
  - Purpose: Infer data format and semantic metadata for user verification
  - Input: `data_summary: DataSummary, user_hints: Optional[str] = None`
  - Output: `{success: bool, format: str, format_evidence: str, schema: Optional[SemanticSchema], error: Optional[str]}`
  - Note: `format` is one of `"pointwise"`, `"pairwise"`, or `"multiway"`; `SemanticSchema = {bigbetter: int, ranking_items: List[str], indicator_col: Optional[str], indicator_values: List[str]}`

- **`validate_data_format`**
  - Purpose: Check if data format/structure can physically run in R script (fixable issues)
  - Input: `file_path: str, schema: SemanticSchema`
  - Output: `{is_ready: bool, fixable: bool, issues: List[str], suggested_fixes: List[str]}`

- **`validate_data_quality`**
  - Purpose: Check if data meets statistical requirements for ranking (unfixable issues)
  - Input: `file_path: str, schema: SemanticSchema`
  - Output: `{is_valid: bool, warnings: List[str], errors: List[str]}`

- **`preprocess_data`**
  - Purpose: Restructure data to R-compatible format (format fixes)
  - Input: `file_path: str, schema: SemanticSchema`
  - Output: `{preprocessed_csv_path: str, transformation_log: List[str], row_count: int, dropped_rows: int}`

#### User Interaction Tool (1)

- **`request_user_confirmation`**
  - Purpose: Explicit user interaction point for schema confirmation
  - Input: `proposed_schema: SemanticSchema, format_result: FormatValidationResult, quality_result: QualityValidationResult`
  - Output: `{confirmed: bool, confirmed_schema: SemanticSchema, user_modifications: List[str]}`

#### Engine Tool (1)

- **`execute_spectral_ranking`**
  - Purpose: Invoke R script for spectral computation
  - Input: `config: EngineConfig`
  - Output: `{success: bool, results: Optional[RankingResults], error: Optional[str], trace: ExecutionTrace}`

#### Analysis Tools (3)

- **`generate_report`**
  - Purpose: Generate structured analysis report
  - Input: `results: RankingResults, session: SessionMemory`
  - Output: `{markdown: str, key_findings: Dict[str, Any], artifacts: List[ArtifactRef], hints: List[HintSpec], citation_blocks: List[CitationBlock]}`

- **`generate_visualizations`**
  - Purpose: Create ranking visualizations
  - Input: `results: RankingResults, viz_types: List[str]`
  - Output: `{plots: List[PlotSpec], errors: List[str]}`
  - Note: `PlotSpec = {type: str, data: dict, config: dict, svg_path: str, block_id: str, caption_plain: str, caption_academic: str, hint_ids: List[str]}`

- **`answer_question`**
  - Purpose: Answer user follow-up questions
  - Input: `question: str, session: SessionMemory, quotes: Optional[List[QuotePayload]]`
  - Output: `{answer: str, supporting_evidence: List[str], used_citation_block_ids: List[str]}`
  - Note: Must work in **all session stages**; uses session.current_results when available, otherwise uses session/data state + literature context (`.agent/literature/spectral_ranking_inferences.md`)

### Data Tools Details

#### Tool: `read_data_file`

```python
def read_data_file(file_path: str) -> ReadDataFileResult:
    """
    Reads uploaded file and returns a lightweight summary for LLM reasoning.
    
    This tool implements JIT (Just-In-Time) context loading - the full dataset
    remains on disk while only essential metadata enters the context window.
    
    Returns:
        ReadDataFileResult containing:
        - success: bool
        - data: Optional[DataSummary]
        - error: Optional[str]
        
        When success=True, data contains a DataSummary with:
        - columns: List of column names
        - sample_rows: First 10 rows as list of dicts
        - row_count: Total number of rows
        - column_types: Dict mapping column names to inferred types
    
    Design Rationale:
        Following Manus Context Engineering: "Use file system as externalized memory.
        Read it only when the information is needed and write it when significant
        state changes."
    """
```

#### Tool: `infer_semantic_schema`

```python
def infer_semantic_schema(
    data_summary: DataSummary, 
    user_hints: Optional[str] = None
) -> SemanticSchemaResult:
    """
    Infers data format and semantic metadata for user verification.
    
    This tool analyzes the data summary to understand both what type of 
    comparison data it represents and what each column means semantically.
    
    Format Detection:
        Identifies the comparison type based on column patterns:
        - Pointwise: Contains Item and Score columns (e.g., ratings data)
        - Pairwise: Contains two item columns and winner/outcome indicator
        - Multiway: Contains rank-ordered columns (Rank_1, Rank_2, ..., Rank_k)
    
    Semantic Inference:
        1. bigbetter: 
           - Priority 1: Semantic keywords (accuracy->1, latency->0)
           - Priority 2: Distribution analysis (0-1 bounded -> 1)
           - Priority 3: User hints if provided
           
        2. ranking_items:
           - For WIDE format: numeric column names are ranking items
           - For LONG format: unique values in item_name column
           
        3. indicator_col (Optional):
           - A categorical column for stratified analysis (e.g., "Task", "Category")
           - Selection Criteria:
             * Must be categorical/string type, not numeric
             * Cardinality: 2-20 unique values preferred (max 50)
             * Semantic Preference: Column names matching keywords (task, category, type, group, class, domain)
             * CRITICAL: Select AT MOST ONE indicator column
           - Returns null if:
             * No categorical columns exist
             * All categorical columns have cardinality < 2 or > 50
             * No semantically meaningful stratification dimension found
           
        4. indicator_values:
           - Unique values within selected indicator column
           - Used for stratified ranking (segmenting results by task/category)
    
    Parameters:
        data_summary: Lightweight data summary from read_data_file
        user_hints: Optional natural language hints from user when re-inferring
                    after rejection (e.g., "the Score column represents accuracy")
    
    Returns:
        SemanticSchemaResult containing:
        - success: bool
        - format: "pointwise" | "pairwise" | "multiway"
        - format_evidence: str explaining the detection reasoning
        - schema: SemanticSchema if successful
        - error: str if failed
    
    User Override:
        All inferred values are presented to user for verification/modification
        via request_user_confirmation before execution.
    """
```

#### Tool: `validate_data_format`

```python
def validate_data_format(
    file_path: str, 
    schema: SemanticSchema
) -> FormatValidationResult:
    """
    Checks if the data structure can physically run in the R script.
    
    This tool validates FORMAT/STRUCTURE compatibility (fixable through preprocessing),
    not statistical validity. It operates BEFORE quality checks because format
    issues must be resolved before meaningful statistical checks can be performed.
    
    Format Checks:
    1. CSV Parsability:
       - Can the file be parsed as a valid CSV/table?
       - Are there encoding issues?
       
    2. Column Availability:
       - Are there at least 2 extractable numeric columns (ranking items)?
       - Can ranking_items from schema be mapped to actual columns?
       
    3. Format Compatibility:
       - Is the data format (wide/long) compatible with R script expectations?
       - Does the structure match the inferred format (pointwise/pairwise/multiway)?
       
    4. Data Type Validity:
       - Are ranking columns numeric or convertible to numeric?
       - Are metadata columns properly identified?
    
    Returns:
        FormatValidationResult containing:
        - is_ready: bool (True if format is R-ready)
        - fixable: bool (True if issues can be fixed by preprocess_data)
        - issues: List[str] (specific format problems found)
        - suggested_fixes: List[str] (preprocessing operations needed)
    
    Design Rationale:
        Separating format from quality checks enables:
        1. Early detection of format mismatches
        2. Automatic format fixes via preprocess_data
        3. Meaningful quality checks only after format is correct
    """
```

#### Tool: `validate_data_quality`

```python
def validate_data_quality(
    file_path: str, 
    schema: SemanticSchema
) -> QualityValidationResult:
    """
    Checks if data meets statistical requirements for spectral ranking.
    
    This tool validates DATA QUALITY and statistical requirements (unfixable issues).
    It operates AFTER format checks because statistical metrics (sparsity,
    connectivity) can only be computed on correctly formatted data.
    
    Quality Checks:
    1. Warnings (Non-blocking):
       - **Sparsity Check**: Warn if M < n*log(n) where M=comparisons, n=items.
         Rationale: Below this threshold, spectral estimation is unstable and statistical inference (CIs) may be less reliable.
       
    2. Errors (Blocking):
       - **Connectivity Check**: Verify comparison graph strong connectivity using networkx.
         Rationale: Disconnected components make global scores mathematically incomparable, preventing a valid global ranking.
       - **Data Integrity**: 
         - Fewer than 2 items present (after preprocessing).
         - Data does not represent valid/logical comparisons.
         - All comparisons involve the same items (no variation).
    
    Returns:
        QualityValidationResult containing:
        - is_valid: bool (True if data meets statistical requirements)
        - warnings: List[str] (non-blocking issues user should be aware of)
        - errors: List[str] (blocking issues that prevent execution)
    
    Design Rationale:
        Quality checks validate data CONTENT and statistical validity, not format.
        These issues cannot be fixed by preprocessing - they require user to provide
        better data or more comparisons.
    """
```

**Theoretical Basis**:
- **Sparsity Threshold ($n \log n$):** Defines the phase transition for graph connectivity and statistical reliability. According to *Assumption 4* and *Theorem 4* in `.agent/literature/spectral_ranking_inferences.md`, this sample complexity is necessary for the spectral estimator to concentrate sufficiently for valid inference (Gaussian multiplier bootstrap).
- **Connectivity:** A necessary condition for Markov chain ergodicity. As per *Section 2.2* in `.agent/literature/spectral_ranking_inferences.md`, strong connectivity ensures a unique stationary distribution; if disconnected, the ranking scores of different components are mathematically incomparable.

#### Tool: `preprocess_data`

```python
def preprocess_data(
    file_path: str, 
    schema: SemanticSchema
) -> PreprocessResult:
    """
    Restructures data to R-compatible format (format fixes only).
    
    This tool operates WITHIN the FORMAT VALIDATION LOOP to fix format issues
    identified by validate_data_format. It performs schema-guided transformations
    to make data physically runnable in the R script.
    
    R Script Input Format Expectation:
        The spectral_ranking.R script expects CSV data in WIDE format where:
        - Each column (except indicator_col) represents an item to rank
        - Each row represents a comparison/observation
        - Cell values are numeric scores (higher = better if bigbetter=1)
        
        The R script internally handles the spectral decomposition to infer
        rankings from these comparison scores. It accepts all three input
        data types (pointwise, pairwise, multiway) as long as they are
        provided in the correct wide-format structure.
    
    Transformation Steps:
    1. Format Restructuring:
       - Long format -> Wide format (pivot item_name column to columns)
       - Multiway -> Wide format normalization (Rank_1..Rank_k to columns)
       - Note: Pairwise data is NOT converted to pointwise; instead it's
         restructured to wide format where the R script applies appropriate
         spectral methods for pairwise comparisons.
       
    2. Column Mapping:
       - Rename columns to match ranking_items from schema
       - Extract numeric columns representing ranking items
       - Preserve indicator_col for segmented analysis
       - Drop non-ranking metadata columns
       
    3. Data Type Conversion:
       - Convert string scores to numeric
       - Standardize item name encoding (UTF-8)
       - Handle missing values (impute or drop based on strategy)
       
    4. Output Generation:
       - Write preprocessed CSV to session temp directory
       - Generate transformation log for audit trail
       - Return new file path for subsequent validation
    
    Returns:
        PreprocessResult containing:
        - preprocessed_csv_path: Path to the transformed CSV file
        - transformation_log: List of applied transformations
        - row_count: Number of rows after preprocessing
        - dropped_rows: Number of rows removed during cleaning
    
    Design Rationale:
        Preprocessing is a FORMAT FIX tool within the validation loop:
        1. Fixes format issues identified by validate_data_format
        2. Schema guides all transformations (what columns are items, etc.)
        3. After preprocessing, data re-enters format check to verify fix
        4. SessionMemory.current_file_path is updated to the new path
    """
```

#### Tool: `request_user_confirmation`

```python
def request_user_confirmation(
    proposed_schema: SemanticSchema,
    format_result: FormatValidationResult,
    quality_result: QualityValidationResult
) -> ConfirmationResult:
    """
    Explicit user interaction point for schema and configuration confirmation.
    
    This tool makes the implicit "[User Interactive Configuration via UI]" 
    step explicit, ensuring the agent knows when to wait for user input.
    
    This tool is called AFTER the DATA VALIDATION LOOP completes successfully,
    meaning data has passed both format and quality checks (or has only
    non-blocking warnings).
    
    Presentation to User:
    1. Inferred Schema Summary:
       - bigbetter: "Higher values are better" or "Lower values are better"
       - ranking_items: List of detected items to rank
       - indicator_col: Selected segmentation column (if any)
       
    2. Validation Status:
       - Warnings: List of non-blocking quality concerns (e.g., sparsity) that the user should acknowledge before proceeding.
       
    3. Modification Options:
       - User can modify any inferred value (bigbetter, ranking_items, etc.)
       - User can change indicator column selection
       - **Advanced Settings**: User can adjust bootstrap iterations (B) and random seed.
    
    Returns:
        ConfirmationResult containing:
        - confirmed: bool (True if user approves, False if user cancels)
        - confirmed_schema: SemanticSchema (original or modified)
        - user_modifications: List of changes made by user
    
    Error Handling:
        If user rejects the schema entirely, the agent should:
        1. Ask clarifying questions about the data
        2. Re-run infer_semantic_schema with user hints
        3. Re-enter validation loop with updated schema
    
    Design Rationale:
        Confirmation happens AFTER validation ensures:
        1. User sees validated, R-ready data structure
        2. User makes informed decision with full validation context
        3. Schema modifications trigger re-validation (loop back)
        4. Audit trail includes user decision point
    """
```

### Engine Tool Details

#### Tool: `execute_spectral_ranking`

```python
def execute_spectral_ranking(config: EngineConfig) -> ExecutionResult:
    """
    Deterministic tool that invokes spectral_ranking.R.
    
    This tool is NOT an LLM call - it executes the R script in a subprocess
    and parses the JSON output.
    
    Config Parameters:
    - csv_path: Path to preprocessed data file
    - bigbetter: 1 (higher is better) or 0 (lower is better)
    - selected_items: Optional filter for specific items
    - selected_indicator_values: Optional filter for indicator segments
    - B: Bootstrap iterations (default 2000)
    - seed: Random seed (default 42)
    
    Returns:
        ExecutionResult containing:
        - success: bool
        - results: RankingResults if successful (theta_hat, ranks, CIs)
        - error: str if failed
        - trace: ExecutionTrace for debugging
    
    CLI Invocation:
        Rscript src/spectral_ranking/spectral_ranking.R \
            --csv {csv_path} --bigbetter {bigbetter} \
            --B {B} --seed {seed} --out {output_dir}
    """
```

### Analysis Tools Details

#### Tool: `generate_report`

```python
def generate_report(
    results: RankingResults, 
    session: SessionMemory
) -> ReportOutput:
    """
    Generate a publication-ready markdown report from RankingResults.

    Implementation: Report narrative and recommendations are generated by LLM
    using session results and the writing contract below.

    Audience-first, single-page progressive disclosure contract (Deep Research style):
    - The entire report must render as ONE continuous page (single scroll) with
      narrative text and figures interleaved.
    - No pagination, no separate appendix pages, and no collapsible sections.
    - Statistical rigor is preserved via inline micro-explanations and icon-triggered
      tooltips/popovers (definitions, assumptions, caveats) that do NOT require
      expanding hidden content.

    Report layout (single-page, in reading order):
    1) Executive Summary (non-technical, above the fold)
       - Top-ranked item(s) and what "best" means in this context
       - Plain-language uncertainty statement (what we can / cannot conclude)
       - Key takeaways and actionable recommendations

    2) Interleaved Results Narrative (technical-lite)
       - Short paragraphs explaining the ranking story, immediately followed by
         the relevant figure/table snippet
       - Inline fields: rank, item, score (theta_hat), uncertainty (e.g., 95% bootstrap CI)
       - Notes on ties / near-ties and practically meaningful gaps
       - Inline "info" icons for: definitions (theta_hat, CI), interpretation rules,
         and common pitfalls (e.g., CI overlap is not a formal test)

    3) Targeted Comparisons (as-needed, still on the same page)
       - Focused comparisons for stakeholder questions (e.g., A vs B)
       - Clear decision rules and caveats (avoid overclaiming)
       - Tooltip/popover micro-notes for formal vs informal evidence language

    4) Methods, Assumptions, and Limitations (academic, concise, inline)
       - A compact methods block written for scholars, placed near the first figure
         that depends on it (not at the end)
       - Mandatory items (kept brief): estimator definition, bootstrap CI recipe,
         CI level, seed, B, filtering rules, data coverage, limitations

    5) Reproducibility (inline, actionable)
       - EngineConfig snapshot and run metadata (timestamps, versions if available)
       - Paths to generated artifacts (tables/figures) and how to regenerate them
       - If full technical detail is needed, link to deterministic artifacts
         (e.g., JSON summaries, SVGs) rather than adding hidden sections.

    Citable quoting (mainstream LLM client behavior):
    - The rendered report must support "select -> click Quote -> insert into composer".
    - Every citable unit (paragraph, bullet cluster, table snippet, figure caption)
      MUST be wrapped with a stable block identifier in the markdown using raw HTML.

      Required wrapper pattern (renderer-friendly):
        <section data-omni-block-id="{block_id}" data-omni-kind="{kind}">
          ...markdown content...
        </section>

      Notes:
      - block_id MUST be stable within the report (deterministic given inputs).
      - kind SHOULD be one of: "summary" | "result" | "comparison" | "method" |
        "limitation" | "repro" | "figure" | "table".
      - Do not hide information behind collapsible sections; tooltips/popovers are
        allowed only for micro-explanations (HintSpec) and do not change the
        single-page requirement.

    Returns:
        ReportOutput with:
        - markdown: str (the report)
        - key_findings: dict (machine-readable highlights for follow-up tools)
        - artifacts: list (references to saved figures/tables, if produced)
        - hints: list (inline micro-explanations for icon tooltips/popovers)
        - citation_blocks: list (structured citable blocks for Quote UX)
    """
```

#### Tool: `generate_visualizations`

```python
def generate_visualizations(
    results: RankingResults, 
    viz_types: List[str]
) -> VisualizationOutput:
    """
    Create publication-ready, deterministic SVG figures from RankingResults.

    Progressive disclosure goal (single-page):
    - Figures must be readable in context, immediately where they are referenced.
    - Each figure includes a short, plain-language caption plus a rigorous,
      publication-style caption.
    - Any technical definitions are delivered via inline icon tooltips/popovers,
      not via separate pages or collapsible appendices.

    Determinism:
    - No LLM calls.
    - Same inputs -> same SVG outputs (subject to matplotlib version).

    Supported viz_types:
    - "ranking_bar": Reader-friendly overview (sorted bars) with uncertainty whiskers
    - "ci_forest": Methods-forward forest plot emphasizing confidence intervals

    Parameters:
        results: RankingResults from execute_spectral_ranking
        viz_types: List of visualization types to generate

    Output conventions (for accessibility and academic style):
    - Use a colorblind-safe palette; never rely on color alone for meaning
    - Use clear axis labels and units; define theta_hat and CI level in captions
    - Include a short, plain-language subtitle or caption per figure
    - Export as SVG for print-quality embedding in the report

    Returns:
        VisualizationOutput containing:
        - plots: List[PlotSpec] where PlotSpec = {
            type: str,           # viz_type name
            data: dict,          # exact data used for plotting (for reproducibility)
            config: dict,        # plot configuration (labels, CI level, palette, etc.)
            svg_path: str,       # path to saved SVG file
            block_id: str,       # stable report block_id for Quote UX (kind="figure")
            caption_plain: str,  # short, non-technical caption for inline reading
            caption_academic: str, # publication-style caption for scholarly readers
            hint_ids: List[str]  # tooltip/popover hint references
          }
        - errors: List[str]      # any viz_types that failed to render

    Storage:
        Save plots under a session-scoped artifact directory to enable later
        retrieval and inclusion in generate_report.
    """
```

#### Tool: `answer_question`

```python
def answer_question(
    question: str,
    session: SessionMemory,
    quotes: Optional[List["QuotePayload"]] = None
) -> AnswerOutput:
    """
    Answers user follow-up questions using session context and spectral knowledge.
    
    This tool MUST be callable in any stage. It accesses:
    - session.current_results (if available) for item-level rank/CI answers
    - session.data_summary / schema / status for pre-run stage-aware answers
    - .agent/literature/spectral_ranking_inferences.md for theory-grounded context
      and reference links (when method/statistical explanation is requested).

    Quote-aware follow-ups (optional):
    - If quotes are provided, prioritize answering the question with respect to the
      quoted text and its referenced block_id(s), while still using session context
      for numerical verification and caveats.
    
    Context Retrieval Strategy:
    1. Result Cache: For ranking/score queries
       - "Is Model A better than B?" -> Compare theta_hat and integer CIs; avoid treating CI overlap as a formal test
       
    2. Data State: For data property queries
       - "How many comparisons?" -> Check session.data_summary
       
    3. Execution Trace: For process queries
       - "Did it converge?" -> Check session.execution_trace

    4. Literature Context: For methodological/statistical interpretation
       - Cite correct paper title + URL from spectral_ranking_inferences references when needed
    
    Knowledge Base:
    - Spectral ranking theory (CI interpretation, significance testing)
    - Domain expertise embedded in system prompt
    
    Returns:
        AnswerOutput containing answer, supporting evidence, and the report block_id(s)
        used to answer (when available).
    """
```

### Type Definitions

All custom types used by OmniRank tools are defined below for implementation clarity:

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataSummary:
    """Lightweight summary of uploaded data file."""
    columns: List[str]
    sample_rows: List[Dict[str, Any]]  # First 10 rows as list of dicts
    row_count: int
    column_types: Dict[str, str]  # column_name -> "numeric" | "categorical" | "datetime"

@dataclass
class ReadDataFileResult:
    """Structured result from read_data_file (robust error handling)."""
    success: bool
    data: Optional[DataSummary] = None
    error: Optional[str] = None

@dataclass
class SemanticSchema:
    """Inferred semantic metadata for R script execution."""
    bigbetter: int  # 1 = higher is better, 0 = lower is better
    ranking_items: List[str]  # Column names or item values to rank
    indicator_col: Optional[str]  # Segmentation column (e.g., "benchmark")
    indicator_values: List[str]  # Unique values in indicator_col

@dataclass
class SemanticSchemaResult:
    """Combined result of format detection and semantic schema inference."""
    success: bool
    format: str  # "pointwise" | "pairwise" | "multiway"
    format_evidence: str  # Explanation of format detection reasoning
    schema: Optional[SemanticSchema]  # Inferred schema if successful
    error: Optional[str] = None  # Error message if failed

@dataclass
class FormatValidationResult:
    """Result of data format validation."""
    is_ready: bool  # True if format is R-ready
    fixable: bool  # True if issues can be fixed by preprocess_data
    issues: List[str]  # Specific format problems found
    suggested_fixes: List[str]  # Preprocessing operations needed

@dataclass
class QualityValidationResult:
    """Result of data quality validation."""
    is_valid: bool  # True if data meets statistical requirements
    warnings: List[str]  # Non-blocking issues
    errors: List[str]  # Blocking issues that prevent execution

@dataclass
class PreprocessResult:
    """Result of data preprocessing transformation."""
    preprocessed_csv_path: str
    transformation_log: List[str]
    row_count: int
    dropped_rows: int

# ─────────────────────────────────────────────────────────────────────────────
# Engine Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EngineConfig:
    """Configuration for spectral ranking R script execution."""
    csv_path: str  # Path to preprocessed data file
    bigbetter: int  # 1 (higher is better) or 0 (lower is better)
    selected_items: Optional[List[str]] = None  # Filter for specific items
    selected_indicator_values: Optional[List[str]] = None  # Filter for segments
    B: int = 2000  # Bootstrap iterations
    seed: int = 42  # Random seed for reproducibility
    r_script_path: str = "src/spectral_ranking/spectral_ranking.R"  # Configurable path

@dataclass
class RankingResults:
    """Results from spectral ranking computation."""
    items: List[str]  # Item names in order
    theta_hat: List[float]  # Estimated scores
    ranks: List[int]  # Inferred ranks (1 = best if bigbetter=1)
    ci_lower: List[float]  # 95% CI lower bounds
    ci_upper: List[float]  # 95% CI upper bounds
    indicator_value: Optional[str] = None  # Which segment this result is for

@dataclass
class ExecutionTrace:
    """Debugging trace for R script execution."""
    command: str  # Full CLI command executed
    stdout: str  # R script stdout
    stderr: str  # R script stderr
    exit_code: int
    duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ExecutionResult:
    """Combined output of spectral ranking engine execution."""
    success: bool
    results: Optional[RankingResults]
    error: Optional[str]
    trace: ExecutionTrace

# ─────────────────────────────────────────────────────────────────────────────
# Session Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    """Record of a single tool invocation."""
    tool_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error: Optional[str] = None

@dataclass
class SessionMemory:
    """
    Session-scoped memory for maintaining analysis context across tool calls.
    """
    # Data State
    original_file_path: Optional[str] = None
    current_file_path: Optional[str] = None  # May differ after preprocessing
    data_summary: Optional[DataSummary] = None
    inferred_schema: Optional[SemanticSchema] = None
    format_validation_result: Optional[FormatValidationResult] = None
    quality_validation_result: Optional[QualityValidationResult] = None
    
    # Execution State
    execution_trace: List[ExecutionTrace] = field(default_factory=list)
    current_results: Optional[RankingResults] = None
    
    # Conversation State
    user_queries: List[str] = field(default_factory=list)
    tool_call_history: List[ToolCall] = field(default_factory=list)

# ─────────────────────────────────────────────────────────────────────────────
# Output Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlotSpec:
    """Specification for a generated visualization."""
    type: str  # viz_type name
    data: Dict[str, Any]  # Data used for plotting
    config: Dict[str, Any]  # Plot configuration
    svg_path: str  # Path to saved SVG file
    block_id: str = ""  # Stable report block_id for Quote UX (kind="figure")
    caption_plain: str = ""  # Short, non-technical caption for inline reading
    caption_academic: str = ""  # Publication-style caption for scholarly readers
    hint_ids: List[str] = field(default_factory=list)  # Tooltip/popover hint references

@dataclass
class ArtifactRef:
    """Reference to a deterministic artifact generated during analysis."""
    kind: str  # e.g., "figure", "table", "json_summary"
    path: str  # Filesystem path or storage key
    title: str = ""  # Human-readable label
    mime_type: str = "text/plain"  # e.g., "image/svg+xml"

@dataclass
class HintSpec:
    """
    Inline micro-explanation attached to an icon in the single-page report.
    The renderer can map hint_id -> tooltip/popover content without hiding sections.
    """
    hint_id: str  # Stable identifier referenced by hint_ids in PlotSpec / report markdown
    title: str  # Short label shown in UI
    body: str  # Tooltip/popover text (keep concise)
    kind: str = "definition"  # "definition" | "assumption" | "caveat" | "method"
    sources: List[str] = field(default_factory=list)  # Optional citations/links

@dataclass
class CitationBlock:
    """
    A citable unit in the single-page report for Quote UX.
    The UI may embed block_id into the DOM and also use this structured form.
    """
    block_id: str  # Stable identifier (deterministic given inputs)
    kind: str  # See data-omni-kind values in generate_report contract
    markdown: str  # The markdown/HTML snippet for this block
    text: str  # Plain-text version (for copying into the composer)
    hint_ids: List[str] = field(default_factory=list)  # Inline hint references
    artifact_paths: List[str] = field(default_factory=list)  # Related SVG/JSON paths

@dataclass
class QuotePayload:
    """
    Payload inserted into the composer when the user clicks Quote on a selection.
    This mirrors mainstream LLM client quote interactions.
    """
    quoted_text: str  # What the user selected (plain text)
    block_id: Optional[str] = None  # Nearest data-omni-block-id, if available
    kind: Optional[str] = None  # Optional: copied from data-omni-kind
    source: str = "report"  # "report" | "user_upload" | "external"

@dataclass
class VisualizationOutput:
    """Output from generate_visualizations tool."""
    plots: List[PlotSpec]
    errors: List[str]  # Any viz_types that failed to render

@dataclass
class ReportOutput:
    """Output from generate_report tool."""
    markdown: str  # Single-page report with interleaved narrative + figure references
    key_findings: Dict[str, Any]  # Machine-readable highlights for follow-up tools
    artifacts: List[ArtifactRef] = field(default_factory=list)  # Deterministic outputs (SVG/JSON/etc.)
    hints: List[HintSpec] = field(default_factory=list)  # Inline micro-explanations for icon tooltips
    citation_blocks: List[CitationBlock] = field(default_factory=list)  # Quote-able blocks

@dataclass
class AnswerOutput:
    """Output from answer_question tool."""
    answer: str
    supporting_evidence: List[str]
    used_citation_block_ids: List[str] = field(default_factory=list)

@dataclass
class ConfirmationResult:
    """Output from request_user_confirmation tool."""
    confirmed: bool
    confirmed_schema: SemanticSchema
    user_modifications: List[str]
    B: int = 2000  # User-confirmed bootstrap iterations
    seed: int = 42  # User-confirmed random seed
```

## OmniRank Agent System Prompt

**Single-Source Constraint**: All prompts used by the OmniRank agent system MUST reside exclusively in ONE file. No prompts shall be scattered across multiple modules, configs, or inline strings. This ensures maintainability, version control clarity, and consistent prompt governance.

## OmniRank Agent Experiments

This section documents the experimental design for validating OmniRank's capabilities. The experiments are structured to verify the contributions stated in this document and demonstrate OmniRank's advantages over generic LLM agents.

```
4 Experiments
├── 4.1 Tool Capability Evaluation
│   ├── 4.1.1 Schema Inference Evaluation
│   └── 4.1.2 Data Validation Pipeline Evaluation
└── 4.2 Comparison with Generic LLM Agents
    ├── 4.2.1 Task Design
    └── 4.2.2 Evaluation Protocol
```

### 4.1 Tool Capability Evaluation

This section evaluates the intelligent capabilities of OmniRank's tools within the Single Agent + Tool Calling architecture.

#### 4.1.1 Schema Inference Evaluation

**Purpose**: Evaluate the `infer_semantic_schema` tool's ability to automatically detect data format and infer semantic metadata.

**Tool Under Test**: `infer_semantic_schema` (see `src/api/tools/infer_semantic_schema.py`)

The tool performs two key functions:
1. **Format Detection**: Classify input data as `pointwise`, `pairwise`, or `multiway`
2. **Schema Inference**: Infer `bigbetter`, `ranking_items`, `indicator_col`, and `indicator_values`

**Test Dataset Design Principles**:

*Format Detection Categories*:
- **Standard**: Clean, unambiguous structures for each format type
- **Ambiguous**: Cases where data characteristics could lead to misclassification (e.g., pairwise data that resembles pointwise)
- **Edge Cases**: Transposed orientations, minimal valid data, boundary conditions
- **Invalid**: Data that should trigger rejection (empty files, non-tabular data)
- **Real-world**: Representative samples from common domains (sports rankings, LLM benchmarks, product comparisons)

*Schema Inference Categories*:
- **BigBetter Clear**: Metrics with unambiguous preference direction (accuracy, error_rate, latency)
- **BigBetter Ambiguous**: Semantically ambiguous column names requiring contextual inference
- **Indicator Present/Absent**: Datasets with and without meaningful stratification dimensions
- **Items Identification**: Various column layouts challenging item detection heuristics

**Ground Truth Labeling**:
Each test dataset must include human-annotated labels:
- `expected_format`: `pointwise` | `pairwise` | `multiway`
- `expected_bigbetter`: `0` | `1`
- `expected_ranking_items`: List of column names
- `expected_indicator_col`: Column name or `null`
- `expected_indicator_values`: List of values or `[]`

**Evaluation Metrics**:

| Component | Metric | Definition |
|-----------|--------|------------|
| Format Detection | Accuracy | Proportion of correctly classified formats |
| Format Detection | Confusion Matrix | Per-format precision/recall/F1 |
| BigBetter | Accuracy | Proportion of correctly inferred preference directions |
| Ranking Items | Jaccard Index | $|Predicted \cap Ground Truth| / |Predicted \cup Ground Truth|$ |
| Indicator Column | Accuracy | Correct column selection (including null detection) |
| Indicator Values | F1 Score | Precision/recall of value enumeration |

**Experimental Procedure**:
1. Prepare test dataset collection with ground truth annotations
2. For each dataset, invoke `infer_semantic_schema` with `user_hints=None`
3. Compare tool output against ground truth
4. Aggregate metrics by category and overall

#### 4.1.2 Data Validation Pipeline Evaluation

**Purpose**: Evaluate the `validate_data_format`, `validate_data_quality`, and `preprocess_data` tools.

**Tools Under Test**:
- `validate_data_format` (see `src/api/tools/validate_data_format.py`): Checks structural readiness
- `validate_data_quality` (see `src/api/tools/validate_data_quality.py`): Checks statistical validity
- `preprocess_data` (see `src/api/tools/preprocess_data.py`): Performs format transformations

**Test Categories**:

| Category | Description | Expected Tool Behavior |
|----------|-------------|------------------------|
| Valid Wide Format | Correctly structured item-score columns | `is_ready=True`, `is_valid=True` |
| Long Format (Fixable) | Item/value rows requiring pivot | `is_ready=False`, `fixable=True` → preprocess → `is_ready=True` |
| Pairwise Long (Fixable) | item_a/item_b/winner format | `is_ready=False`, `fixable=True` → preprocess → `is_ready=True` |
| Non-Numeric Columns | Ranking columns with string values | `is_ready=False` with appropriate issue message |
| Sparse Data | $M < n \log n$ comparisons | `is_valid=True` with sparsity warning |
| Disconnected Graph | Multiple graph components | `is_valid=False` with connectivity error |
| Insufficient Items | Fewer than 2 ranking items | `is_valid=False` with item count error |

**Evaluation Metrics**:

| Tool | Metric |
|------|--------|
| `validate_data_format` | Issue detection precision/recall |
| `validate_data_format` | Fixable classification accuracy |
| `validate_data_quality` | Warning trigger accuracy (sparsity) |
| `validate_data_quality` | Error trigger accuracy (connectivity, item count) |
| `preprocess_data` | Transformation success rate |
| `preprocess_data` | Data integrity (no information loss) |

**Experimental Procedure**:
1. Prepare test datasets covering each category
2. Run validation tools and record outputs
3. Compare against expected behavior annotations
4. For fixable cases, verify preprocessing resolves issues

### 4.2 Comparison with Generic LLM Agents

**Purpose**: Demonstrate that OmniRank's specialized architecture outperforms generic LLM agents on knowledge-intensive ranking tasks requiring spectral ranking expertise.

#### 4.2.1 Task Design

**Baseline Agents**:

| Agent | Description | Why Included |
|-------|-------------|--------------|
| **gpt-5-mini (raw)** | Same base model as OmniRank, no specialized tools | Ablation: isolate tool contribution |
| **gpt-5 (raw)** | More capable model, no specialized tools | Test if specialized tools on weaker model outperform stronger model without tools |
| **[TBD] Other Ranking Systems** | Existing ranking-related agents or systems | Domain comparison (to be determined through literature review) |

**Task Suite**:

| Task ID | Task Description | Required Domain Knowledge |
|---------|------------------|---------------------------|
| T1 | Given pairwise comparison data, compute spectral ranking scores | Hypergraph construction, transition matrix, stationary distribution |
| T2 | Compute bootstrap confidence intervals for ranking scores | Gaussian multiplier bootstrap methodology |
| T3 | Answer: "Is Model A significantly better than Model B at 95% confidence?" | CI interpretation, statistical inference |
| T4 | Handle dataset with disconnected comparison graph | Graph connectivity theory, component-wise ranking |
| T5 | Provide guidance for sparse dataset ($M \ll n \log n$) | Sample complexity theory, sparsity implications |
| T6 | Infer semantic schema from ambiguous LLM benchmark data | Semantic understanding, format detection |

**Task Presentation**:
- Each task is presented as a natural language prompt with attached CSV data
- No hints about spectral ranking methodology are provided
- Agents must independently determine appropriate methodology

#### 4.2.2 Evaluation Protocol

**Evaluation Dimensions**:

| Dimension | Weight | Criteria |
|-----------|--------|----------|
| Methodology Selection | 30% | Did the agent choose appropriate statistical method? |
| Numerical Correctness | 30% | Are computed scores/CIs numerically accurate? |
| Interpretation Quality | 20% | Are conclusions and recommendations statistically valid? |
| Error Handling | 20% | Did the agent correctly identify and report data issues? |

**Scoring Rubric** (per task):

| Score | Description |
|-------|-------------|
| 1.0 | Fully correct methodology, accurate results, valid interpretation |
| 0.75 | Correct methodology, minor numerical errors or incomplete interpretation |
| 0.5 | Partially correct approach, significant errors in results or interpretation |
| 0.25 | Attempted relevant approach but fundamentally flawed execution |
| 0.0 | Wrong methodology, no meaningful attempt, or critical errors |

**Expected Outcome Hypothesis**:
- OmniRank should achieve near-perfect scores on all tasks due to specialized tool design
- Generic LLMs (even gpt-5) are expected to struggle with spectral ranking-specific methodology
- The comparison will demonstrate that domain-specific tools compensate for model capability differences

**Failure Mode Documentation**:
For each task, document observed failure modes of baseline agents:
- What methodology did they attempt?
- Where did the approach break down?
- What domain knowledge was missing?

### Experiments Implementation Notes

**Test Data Requirements**:
- All test datasets must have human-verified ground truth annotations
- Datasets should cover realistic variation in real-world data characteristics
- Edge cases must be systematically included

**Reproducibility Requirements**:
- Fixed random seed (default: 42) for any stochastic components
- All test datasets and evaluation scripts version-controlled
- Baseline agent API calls should be logged for reproducibility
