# OmniRank: A Large-Language-Model Agent Platform for Statistically Rigorous Ranking Inference from Arbitrary Multiway Comparisons

## Abstract

Spectral ranking inferences provide a minimax optimal solution for analyzing multiway comparison data, which could achieve the same asymptotic efficiency as the Maximum Likelihood Estimation (MLE) while providing extra uncertainty quantifications. However, the steep learning curve of linear algebra-based implementations restricts their utility to a small circle of statisticians. In this study, we present OmniRank, an agentic framework that synergizes reasoning capabilities of Large Language Models (LLMs) with the mathematical rigor of spectral ranking inference. Unlike standard LLMs that are prone to hallucinations in arithmetic tasks, OmniRank decouples instruction following from computation: an LLM controller parses user queries and data, delegating the rigorous inference to a specialized Spectral Calculation Engine. Case study results on both synthetic and real-world datasets confirm that OmniRank achieves precise ranking recovery matching established statistical packages. By providing an interactive, no-code interface for spectral ranking, OmniRank democratizes advanced ranking methods and uncertainty inference for domain experts across social and natural sciences.

## 1 Introduction

Ranking inferences from comparison data are ubiquitous in scientific inquiry and modern applications, ranging from identifying optimal treatments in clinical trials and ranking biological stimuli to evaluating the relative strength of competitors in sports and gaming [1, 2]. While classical frameworks like the Bradley-Terry-Luce model have successfully handled pairwise comparisons, real-world data increasingly manifest as multiway comparisons, where multiple items are compared simultaneously—such as in horse races, multi-player online games, or top-k choice data in econometrics [3, 4]. Unlike pairwise data, multiway comparisons involve hyperedges of heterogeneous sizes, creating complex dependency structures that defy simple aggregation. Although the Plackett-Luce model offers a probabilistic foundation for such data, its reliance on Maximum Likelihood Estimation (MLE) faces significant challenges: the likelihood function can be non-convex, and the computational burden becomes prohibitive as the number of items (n) grows, often requiring O(N^3) complexity for precise inference [5, 6].

To overcome these computational and statistical barriers, recent theoretical breakthroughs have established spectral ranking inferences based on general multiway comparisons as a superior alternative. By constructing a comparison graph where items are nodes and multiway comparisons form hyperedges, these methods utilize the stationary distribution of a random walk on the hypergraph (or the eigenvectors of the hypergraph Laplacian) to recover latent preference scores [7]. Fan et al. demonstrated that this spectral approach achieves minimax optimal statistical rates comparable to MLE but with significantly greater computational efficiency, even under heterogeneous sampling conditions where hyperedge sizes vary dramatically [8]. Despite this theoretical elegance, the practical application of spectral ranking remains confined to a small circle of statisticians. The implementation requires rigorous handling of sparse hypergraph adjacency matrices and complex linear algebra operations, creating a steep technical barrier for domain experts—such as sociologists or biologists—who possess rich multiway data but lack the coding expertise to implement these specialized spectral algorithms [9].

Large Language Models (LLMs) have emerged as potential intermediaries to democratize such advanced analytical tools. Models like GPT-4 have shown impressive capabilities in code generation and logical reasoning [10, 11]. However, standard LLMs inherently struggle with rigorous mathematical execution; they are prone to "hallucinations" when performing arithmetic or executing specific algorithms mentally, and they lack the native ability to process large-scale structured data (e.g., adjacency matrices) directly within their context window [12]. Consequently, the current wave of "AI Agents" has shifted towards a tool-use paradigm, where the LLM acts as a controller that delegates specific tasks to external computational tools [13, 14]. While agents have been developed for chemical synthesis [15] and gene analysis [16], there is currently no dedicated framework that bridges the gap between the sophisticated mathematics of spectral ranking and the intuitive needs of non-technical users.

Here, we introduce OmniRank, a novel web-based agentic framework that democratizes access to spectral ranking inferences. The architecture consists of two synergistic components: an LLM Agent that interprets user’s natural language requests and raw data uploads (e.g., “Rank these polygenic risk scores for breast cancer using their comparative AUC performance across the uploaded validation cohorts.”), and a Spectral Calculation Engine that executes the hypergraph construction and eigenvector computations. The results are then rendered through an interactive visualization dashboard, allowing users to explore ranking confidence intervals and topology without writing a single line of code. By decoupling the complex spectral inference (Backend) from the user interaction (Frontend), we ensure that the mathematical precision of the underlying theory is preserved while maximizing accessibility.

We validated the efficacy of OmniRank through both theoretical benchmarking and real-world application scenarios. To assess the fidelity of our agent-driven pipeline, we compared its output against standard R implementations of spectral ranking on synthetic datasets with varying heterogeneity in comparison sizes (k). Furthermore, we demonstrate the tool’s practical utility by applying it to a real-world LLMs dataset, where the agent successfully parsed unstructured match results and produced rankings consistent with ground-truth outcomes. Our results show that by combining the reasoning power of LLMs with the mathematical rigor of spectral graph theory, we can effectively lower the barrier to entry for advanced statistical ranking, enabling broader application across diverse scientific fields.

## 2 Background and Related Works

The development of OmniRank draws upon two distinct research traditions: statistical methods for ranking inference from comparison data, and the emerging paradigm of LLM-based agents for scientific computing. This section reviews the relevant literature in both areas and positions OmniRank within the broader context of efforts to democratize advanced statistical methods.

### 2.1 Statistical Methods for Ranking from Comparison Data

The problem of inferring rankings from comparison data has a rich history spanning statistics, economics, and machine learning. The Bradley-Terry model [48] and its multiway extension, the Plackett-Luce model [46], have served as foundational frameworks for estimating latent preference scores from pairwise and multiway comparisons, respectively. These parametric models assume that comparison outcomes follow a logistic or multinomial logit distribution governed by item-specific quality parameters.

Classical inference for these models relies on Maximum Likelihood Estimation (MLE), which Hunter [4] showed can be efficiently computed via MM algorithms for the Bradley-Terry case. However, MLE approaches face significant computational challenges as the number of items grows: the likelihood function may be non-convex, and iterative optimization can require substantial computation time [5, 6]. These limitations have motivated the development of computationally efficient alternatives.

Spectral methods have emerged as a compelling solution to these computational bottlenecks. Rather than optimizing a likelihood function, spectral approaches construct a comparison graph where items correspond to nodes and comparisons induce edges, then extract rankings from the eigenvectors of an associated matrix [49]. Negahban et al. [7] demonstrated that spectral methods achieve statistically consistent estimates under the Bradley-Terry model with computational complexity dominated by a single eigenvalue computation. Shah and Wainwright [19] established that even simple counting-based algorithms like Borda count can achieve minimax optimal rates for pairwise ranking under certain conditions.

The theoretical understanding of spectral ranking has advanced considerably in recent years. Chen et al. [38] proved that spectral methods achieve the same optimal rates as regularized MLE for top-$K$ ranking problems while offering superior computational efficiency. Fan et al. [8] extended these results to the general multiway comparison setting, establishing minimax optimality of spectral methods for heterogeneous hypergraph structures. Despite these theoretical advances, the practical implementation of spectral ranking methods remains technically demanding, requiring careful handling of hypergraph Laplacians, eigenvector perturbation bounds, and bootstrap-based uncertainty quantification [41, 52].

### 2.2 Large Language Models as Scientific Agents

The rapid progress of large language models has opened new possibilities for automating scientific workflows [20]. Contemporary LLMs demonstrate impressive capabilities in code generation, logical reasoning, and natural language understanding, prompting researchers to explore their potential as "agents" capable of executing complex analytical tasks [21, 44]. In the data science domain, LLM-based systems have been deployed to automate data preprocessing, model selection, and result interpretation [22].

Several recent works have demonstrated the utility of LLM agents in specialized scientific domains. Boiko et al. [15] presented Coscientist, an agentic framework capable of autonomous chemical synthesis by integrating LLM reasoning with laboratory automation. Similarly, GeneAgent [16] employs self-verification mechanisms for gene-set analysis, while other systems have addressed clinical prediction [23] and materials discovery [24]. These applications share a common architectural pattern: the LLM serves as a cognitive controller that interprets user intent and orchestrates domain-specific computational tools.

A persistent challenge in deploying LLMs for quantitative analysis is their tendency to produce "hallucinations"---outputs that appear plausible but are factually incorrect or computationally erroneous [32]. This limitation is particularly acute for mathematical operations: LLMs can generate syntactically correct formulas that yield numerically wrong results, a failure mode that undermines their reliability for statistical inference [12]. The hallucination problem has motivated the tool-use paradigm [13], where LLMs delegate precise computations to external engines rather than attempting to execute algorithms internally.

The tool-use paradigm represents a fundamental shift in how LLMs are deployed for scientific computing. Systems like Toolformer [13] and TaskMatrix [14] demonstrate that LLMs can learn to invoke appropriate tools based on task requirements. However, most existing frameworks target general-purpose computation; specialized statistical methods like spectral ranking inference have received limited attention. OmniRank addresses this gap by integrating LLM-based data interpretation with a purpose-built spectral computation engine.

### 2.3 Multi-Agent Architectures and Reliability

The transition from single-prompt LLMs to multi-agent collaborative systems has significantly enhanced the reliability of AI-driven analysis. Multi-agent architectures partition complex tasks among specialized agents, each with defined responsibilities and interaction protocols [17, 25]. This modular design facilitates error diagnosis, enables iterative refinement, and supports human intervention when automated processes fail.

MetaGPT [35] exemplifies the multi-agent approach, organizing LLM agents into roles analogous to a software development team (product manager, architect, engineer, QA). While effective for code generation tasks, such heavyweight frameworks incur substantial token costs and may be overly complex for statistical analysis workflows [26]. LAMBDA [26] demonstrated that a simpler two-agent architecture---a "programmer" for code generation and an "inspector" for error diagnosis---can achieve reliable performance with reduced overhead. This programmer-inspector paradigm has proven effective for maintaining code accuracy compared to end-to-end approaches [37].

A critical insight from recent work is that decoupling "reasoning" from "execution" improves reliability for knowledge-intensive tasks [42]. When LLMs attempt to execute statistical algorithms directly, they may produce plausible-looking but incorrect results; delegating computation to verified engines eliminates this failure mode. For spectral ranking inference specifically, this decoupling is essential because correctness depends on strict adherence to graph-theoretic properties---connectivity requirements, sparsity thresholds, and eigenvector convergence---that LLMs cannot verify internally with sufficient precision [8, 38].

OmniRank instantiates these architectural principles in the context of ranking inference. By combining LLM-based semantic understanding (data format recognition, schema inference, result interpretation) with deterministic spectral computation (hypergraph construction, eigenvector extraction, bootstrap confidence intervals), OmniRank achieves both accessibility and statistical rigor. The system's human-in-the-loop design [27] allows domain experts to verify inferred parameters and intervene when automated processes misalign with their analytical goals, addressing a known limitation of fully automated agent systems.

## 3 Methodology

This section presents the methodological framework of OmniRank, an agentic system that bridges large language model reasoning with rigorous spectral ranking inference. We first provide an architectural overview of the single-agent tool-calling design (Section 3.1), then detail the three pipeline phases: data processing and schema inference (Section 3.2), interactive configuration and spectral computation (Section 3.3), and result synthesis with user interaction (Section 3.4). We subsequently describe prompt engineering strategies (Section 3.5) and user interface design (Section 3.6).

### 3.1 Overview

OmniRank employs a single-agent architecture with tool calling, designed to separate semantic understanding from mathematical computation. This architectural principle, termed "decoupled reasoning," addresses a fundamental limitation of current LLMs---their propensity for hallucination in arithmetic and algorithmic tasks [32]---and has proven effective in recent scientific agent systems [33, 34].

The system centers on a single LLM agent operating within one context window, equipped with a registry of ten specialized tools organized into four categories:

- **Data Tools** (5): `read_data_file`, `infer_semantic_schema`, `validate_data_format`, `validate_data_quality`, and `preprocess_data`---responsible for data ingestion, semantic understanding, and validation against spectral ranking requirements.
- **User Interaction Tool** (1): `request_user_confirmation`---an explicit checkpoint for human-in-the-loop parameter verification.
- **Engine Tool** (1): `execute_spectral_ranking`---a deterministic tool that invokes the spectral computation engine in an isolated subprocess.
- **Analysis Tools** (3): `generate_report`, `generate_visualizations`, and `answer_question`---responsible for result interpretation, visualization, and interactive follow-up.

The agent orchestrates these tools through a fixed three-phase pipeline: (1) data processing and schema inference, (2) interactive configuration and spectral computation, and (3) result synthesis and user interaction. Unlike multi-agent architectures that incur coordination overhead [35], this single-agent design keeps all reasoning within one context window, reducing token cost and eliminating inter-agent communication failures. The tool-calling mechanism ensures that precise numerical computation is delegated to verified deterministic tools rather than attempted by the LLM internally.

[Figure: OmniRank system architecture] illustrates the system architecture. The agent receives user requests, selects appropriate tools based on the current pipeline phase, and synthesizes tool outputs into coherent responses. Algorithm 1 formalizes this process.

**[Figure: OmniRank system architecture]** OmniRank employs a single LLM agent with a registry of ten tools. The agent orchestrates a three-phase pipeline: data processing (five data tools), computation (one engine tool with human confirmation), and output generation (three analysis tools). All reasoning occurs within a single context window.

**Algorithm 1** OmniRank Pipeline

**Input:** Dataset $\mathcal{D}$ uploaded by user
**Output:** Ranking results $\mathcal{R}$ with confidence intervals; analysis report

*Phase 1: Data Processing and Schema Inference*
1: $\texttt{summary} \leftarrow \texttt{call}(\texttt{read\_data\_file}, \mathcal{D})$
2: $\mathcal{S} \leftarrow \texttt{call}(\texttt{infer\_semantic\_schema}, \texttt{summary})$
3: **repeat** $\triangleright$ Format validation loop
4:     $\mathcal{V}_f \leftarrow \texttt{call}(\texttt{validate\_data\_format}, \mathcal{D}, \mathcal{S})$
5:     **if** $\mathcal{V}_f.\texttt{fixable}$ **then** $\mathcal{D} \leftarrow \texttt{call}(\texttt{preprocess\_data}, \mathcal{D}, \mathcal{S})$
6: **until** $\mathcal{V}_f.\texttt{is\_ready}$ or not $\mathcal{V}_f.\texttt{fixable}$
7: $\mathcal{V}_q \leftarrow \texttt{call}(\texttt{validate\_data\_quality}, \mathcal{D}, \mathcal{S})$

*Phase 2: Interactive Configuration and Computation*
8: $(\mathcal{S}_c, B, \texttt{seed}) \leftarrow \texttt{call}(\texttt{request\_user\_confirmation}, \mathcal{S}, \mathcal{V}_f, \mathcal{V}_q)$
9: $\mathcal{R} \leftarrow \texttt{call}(\texttt{execute\_spectral\_ranking}, \mathcal{D}, \mathcal{S}_c, B, \texttt{seed})$

*Phase 3: Result Synthesis and User Interaction*
10: $\texttt{plots} \leftarrow \texttt{call}(\texttt{generate\_visualizations}, \mathcal{R})$
11: $\texttt{report} \leftarrow \texttt{call}(\texttt{generate\_report}, \mathcal{R}, \texttt{plots})$
12: **loop** $\triangleright$ Interactive Q&A
13:     $\texttt{answer} \leftarrow \texttt{call}(\texttt{answer\_question}, \texttt{query}, \mathcal{R}, \texttt{report})$
14: **end loop**

This tool-calling pipeline instantiates the "programmer-inspector" paradigm [35] at the tool level: the LLM agent handles reasoning and orchestration, while deterministic tools handle computation and validation. By delegating precise numerical computation to verified tools, OmniRank achieves both accessibility and mathematical rigor.

### 3.2 Data Processing and Schema Inference

The data processing phase serves as the intelligent interface between raw user data and the spectral computation engine. Unlike generic data analysis agents that rely solely on LLM code generation [36], OmniRank employs a hybrid approach: LLM-based semantic reasoning for schema inference combined with deterministic validation rules grounded in spectral ranking theory [37].

#### 3.2.1 Format Recognition and Validation

The agent automatically identifies the structural format of uploaded comparison data and validates its suitability for spectral ranking analysis. We support three canonical formats that encompass the majority of real-world comparison data:

- **Pointwise Format**: Performance metrics for each item across evaluation contexts (e.g., model accuracy on different benchmark tasks).
- **Pairwise Format**: Direct head-to-head comparison outcomes between item pairs (e.g., tournament match results).
- **Multiway Format**: Ranking or selection outcomes from choice sets of arbitrary size (e.g., top-$k$ selections from candidate pools).

Format recognition employs a rule-based classifier augmented with LLM-based disambiguation for edge cases. The agent examines column structure, data types, and semantic patterns to determine the appropriate format, then applies format-specific transformation rules to construct the comparison graph required by the spectral engine.

Following format recognition, the agent performs validation against theoretical requirements established in the spectral ranking literature [8, 38]. Three categories of validation feedback are provided:

**Sparsity Assessment.** The agent evaluates whether the comparison count $M$ satisfies the sample complexity bound $M \geq cn\log n$ for some constant $c > 0$, where $n$ denotes the number of items. This threshold, analogous to the coupon collector bound, represents the minimum sample size required for consistent spectral estimation [38]. When $M < n\log n$, the agent issues a warning indicating that ranking estimates may exhibit elevated variance.

**Connectivity Verification.** Global ranking requires the comparison graph to form a connected component. The agent employs standard graph algorithms to detect disconnected subgraphs. When the graph is disjoint, the agent notifies users that rankings can only be computed within connected components and identifies the largest connected subgraph for analysis.

**Data Integrity Checks.** The agent verifies the presence of required columns, ensures a minimum of two rankable items, and confirms that comparison outcomes are properly encoded. Data failing these checks is rejected with explanatory feedback generated through LLM-based natural language synthesis.

This tiered validation approach, illustrated in [Figure: Data validation workflow], ensures that users receive actionable feedback about data limitations while permitting valid exploratory analyses on imperfect datasets.

**[Figure: Data validation workflow]** The flowchart depicts the hierarchical validation process: critical errors block execution, warnings inform users of theoretical limitations, and valid data proceeds to schema inference.

#### 3.2.2 Semantic Schema Inference

Beyond structural validation, the agent infers the semantic meaning of data components to enable flexible downstream analysis. This capability distinguishes OmniRank from traditional statistical software that requires explicit parameter specification.

**Preference Direction Inference.** The agent determines whether higher metric values indicate superior performance (e.g., accuracy, win rate) or inferior performance (e.g., latency, error rate). This inference combines lexical analysis of column names with distributional properties of the data. For instance, columns containing terms such as "accuracy" or "score" suggest a higher-is-better interpretation, while "time" or "error" suggest the opposite.

**Entity and Indicator Extraction.** The agent identifies:
- *Ranking Items*: The entities to be ranked (e.g., model names, player identifiers).
- *Ranking Indicators*: Categorical dimensions that partition comparisons into semantically meaningful subgroups (e.g., task categories, evaluation conditions).

When multiple potential indicator columns exist, the agent selects at most one to maintain analytical focus, prioritizing columns with moderate cardinality and clear semantic interpretation.

This metadata extraction enables the subsequent configuration phase to present users with intuitive options, allowing customized analysis without requiring statistical expertise.

### 3.3 Interactive Configuration and Computation

This phase manages the transition from inferred schema to statistical computation through human-in-the-loop verification and deterministic engine execution. The design reflects the "tool-use" paradigm in agentic AI, where LLMs serve as cognitive controllers while delegating precise computations to specialized tools [39, 40].

#### 3.3.1 Human-in-the-Loop Configuration

The agent presents inferred parameters through an interactive configuration interface, enabling users to verify and adjust settings before computation. Configurable parameters include:

- **Preference Direction**: Users confirm or override the inferred interpretation of metric values.
- **Item Selection**: Users may restrict analysis to a subset of items.
- **Indicator Selection**: Users select which indicator values to include in the analysis.
- **Statistical Parameters**: Advanced users may configure bootstrap iterations (default: 2,000) and random seed (default: 42) for reproducibility.

This explicit confirmation checkpoint addresses a known limitation of fully automated agent systems: misalignment between inferred parameters and user intent [35]. By requiring human verification, OmniRank ensures that final analyses reflect user requirements while maintaining a complete audit trail of parameter decisions.

#### 3.3.2 Spectral Ranking Inference

The mathematical foundation of OmniRank's computation engine rests on spectral methods for ranking inference from comparison data. We summarize the key elements here; full theoretical treatment including minimax optimality proofs is provided in Fan et al. [8].

Consider $n$ items to be ranked based on comparison outcomes. We model preferences through the Plackett-Luce framework [46, 47], parameterizing item quality by $\boldsymbol{\theta}^* = (\theta_1^*, \ldots, \theta_n^*)^\top$ such that for any choice set $A \subseteq [n]$ and item $i \in A$:

$$P(i \text{ wins among } A) = \frac{e^{\theta_i^*}}{\sum_{k \in A} e^{\theta_k^*}}$$

This model encompasses the Bradley-Terry-Luce model for pairwise comparisons as a special case when $|A| = 2$ [48]. The observed data consist of $L$ comparisons $\{(c_l, A_l)\}_{l=1}^L$, where $A_l$ is the choice set and $c_l \in A_l$ denotes the winner, accommodating heterogeneous comparison structures where choice sets may vary in size [8].

The spectral approach constructs a Markov chain over items whose stationary distribution reflects latent preferences [49, 50]. Define the transition matrix $\mathbf{P}$ with entries:

$$P_{ij} = \frac{1}{d_i} \sum_{l \in W_j \cap L_i} \frac{1}{f(A_l)}$$

where $W_j$ indexes comparisons won by $j$, $L_i$ indexes comparisons lost by $i$, $d_i$ is a normalizing constant, and $f(A_l)$ is a weighting function. The stationary distribution $\hat{\boldsymbol{\pi}}$, obtained as the leading eigenvector of $\mathbf{P}^\top$, yields preference score estimates via the log-transformation $\tilde{\theta}_i = \log \hat{\pi}_i - n^{-1} \sum_{k} \log \hat{\pi}_k$.

Confidence intervals are constructed using the Gaussian multiplier bootstrap [52, 53], which approximates the sampling distribution of $\hat{\boldsymbol{\theta}} - \boldsymbol{\theta}^*$ without parametric assumptions on the comparison process. This enables rigorous uncertainty quantification for ranking conclusions.

#### 3.3.3 Engine Execution

Upon user confirmation, the agent invokes the spectral engine as a deterministic subprocess. The engine applies the spectral estimator with uniform weighting $f(A_l) = |A_l|$ to obtain consistent preference score estimates $\hat{\boldsymbol{\theta}}$ that achieve minimax optimal rates [8]. The execution workflow consists of:

1. **Parameter Preparation**: Constructing the R script command with validated parameters (data path, preference direction, bootstrap iterations, random seed).

2. **Engine Invocation**: Executing `spectral_ranking.R` in an isolated subprocess with timeout protection.

3. **Output Parsing**: Processing the JSON output to extract preference scores ($\hat{\theta}_i$), rankings, and 95% bootstrap confidence intervals.

4. **Trace Logging**: Recording execution parameters and results in session memory for potential error diagnosis.

This deterministic workflow ensures reliable and reproducible ranking computations, fully isolated from the stochastic behavior of the LLM agent.

### 3.4 Result Synthesis and User Interaction

The result synthesis phase transforms computational outputs into interpretable results and supports ongoing user interaction. This phase addresses a critical gap in statistical software: the translation of numerical outputs into actionable insights accessible to domain experts without statistical training.

#### 3.4.1 Report and Visualization Generation

Upon receiving ranking results from the engine, the agent synthesizes comprehensive analysis reports through LLM-based natural language generation. Reports include:

- **Executive Summary**: Key findings highlighting top-ranked items and notable patterns.
- **Detailed Rankings**: Tabular presentation of ranks, preference scores, and confidence intervals with statistical significance indicators.
- **Methodology Notes**: Explanation of the spectral approach and validation outcomes.
- **Domain-Specific Insights**: Contextual interpretation tailored to the data domain, leveraging the semantic schema inferred during data processing.

The agent generates a complementary suite of deterministic visualizations:

1. *Rank Plots*: Forest plots displaying point estimates with confidence interval error bars, enabling visual assessment of ranking uncertainty [41].
2. *Comparison Heatmaps*: Matrix visualizations of pairwise win rates revealing competitive structure among items.
3. *Score Distributions*: Density plots of estimated preference parameters $\hat{\theta}_i$ illustrating the separation between items.

These outputs are rendered in both interactive web formats and exportable static formats (PDF, PNG) suitable for publication.

#### 3.4.2 Interactive Question-Answering

The agent supports follow-up queries through a conversational interface, enabling users to explore results without restarting the analysis. This capability is implemented through a session memory architecture comprising three components:

- **Data State**: Current schema, validation results, and configuration parameters.
- **Execution Trace**: Log of computation invocations and intermediate results for error diagnosis.
- **Conversation Context**: History of user queries and agent responses enabling contextual follow-up.

The agent interprets queries by combining session context with domain knowledge embedded in its system prompt. For example, when a user asks "Is model A significantly better than model B?", the agent retrieves the relevant confidence intervals and applies the non-overlapping confidence interval heuristic to provide a statistically grounded response.

This retrieval-augmented generation approach [42] ensures responses are grounded in computed results rather than hallucinated, addressing a known failure mode of vanilla LLM applications to quantitative domains [32].

### 3.5 Prompt Engineering and Knowledge Integration

We adopt structured system prompts following established practices in LLM agent design [43, 44]. The agent's prompt comprises three layers: role specification, operational constraints, and domain knowledge.

**Role Specification.** Defines the agent's identity and primary responsibilities as a statistical analyst specializing in comparison data formats and ranking inference.

**Operational Constraints.** Specifies output formats, tool invocation protocols, and error handling procedures. These constraints ensure consistent behavior across diverse inputs and enable reliable orchestration of the ten-tool registry.

**Knowledge Layer.** Embeds domain expertise directly into the prompt, enabling expert-level reasoning without requiring fine-tuning. The knowledge layer includes format recognition rules, validation thresholds derived from spectral ranking theory (e.g., the $n\log n$ sparsity threshold), confidence interval interpretation guidelines, and ranking diagnostics.

This knowledge integration approach, illustrated in [Figure: System prompt structure], follows the in-context learning paradigm [45] that has proven effective for knowledge-intensive tasks without model modification. Unlike retrieval-augmented approaches that dynamically fetch external knowledge, our static knowledge layer provides deterministic access to the complete domain context required for spectral ranking analysis.

**[Figure: System prompt structure]** The prompt comprises role specification, format recognition rules, validation thresholds derived from spectral ranking theory, tool invocation protocols, and output format constraints.

### 3.6 User Interface

OmniRank provides a web-based conversational interface designed for accessibility across user expertise levels. The interface guides users through a three-stage workflow:

**Stage 1: Data Upload and Analysis.** Users upload comparison data in standard formats (CSV, Excel). The agent processes the upload, displaying format recognition results, validation outcomes, and inferred schema parameters in an organized panel.

**Stage 2: Interactive Configuration.** The interface presents inferred settings in a visual control panel where users can confirm or modify preference direction, select items and indicator values, and configure advanced statistical parameters. This stage ensures alignment between system inference and user intent before computation proceeds.

**Stage 3: Results and Exploration.** Upon computation completion, the interface displays ranking results with interactive visualizations and a natural language summary. A chat panel enables follow-up queries such as "Which items have statistically indistinguishable rankings?" or "What would change if we excluded item X?"

[Figure: OmniRank user interface] presents interface screenshots illustrating each stage.

**[Figure: OmniRank user interface]** Panel (a) shows the data upload and schema inference display; panel (b) shows the interactive configuration panel; panel (c) shows the results dashboard with visualizations and chat interface.

## 4 Experiments

We evaluate OmniRank through two complementary experimental studies. The first (Section 4.1) assesses the intelligent capabilities of OmniRank's data processing tools, focusing on the accuracy of automatic schema inference and the reliability of the data validation pipeline. These evaluations quantify the system's ability to correctly interpret heterogeneous comparison data formats without manual configuration---a prerequisite for accessible ranking analysis. The second study (Section 4.2) compares OmniRank's end-to-end performance against generic LLM agents on a suite of ranking tasks that require spectral ranking domain knowledge. By contrasting OmniRank (which delegates computation to specialized tools) with raw LLMs of equal or greater capability (which must reason about ranking methodology internally), this comparison isolates the contribution of the tool-calling architecture. Real-world case studies demonstrating practical utility on PRS benchmarking and LLM arena datasets are presented in Section 5.

### 4.1 Tool Capability Evaluation

Among OmniRank's ten tools, five are deterministic (the R spectral engine, visualization generator, report generator, user confirmation handler, and question-answering module). Their correctness is guaranteed by construction and verified through unit tests. The remaining five---`read_data_file`, `infer_semantic_schema`, `validate_data_format`, `validate_data_quality`, and `preprocess_data`---rely on LLM-based reasoning to interpret heterogeneous input data. We focus our evaluation on these data tools, as they constitute the critical interface between raw user data and the spectral computation engine. All experiments use gpt-5-mini (temperature = 0) for reproducibility.

#### 4.1.1 Schema Inference Evaluation

**Evaluation setup.** The `infer_semantic_schema` tool must correctly resolve two tasks: (i) detecting whether the input represents pairwise or multiway comparison data, and (ii) extracting four semantic fields---comparison direction (`bigbetter`), ranking items, indicator column, and indicator values. We constructed a benchmark of [48] datasets spanning five format categories (standard pairwise, standard multiway, transposed, indicator-embedded, and ambiguous/adversarial) and four schema complexity levels (single-indicator, multi-indicator, implicit-indicator, and no-indicator). For each dataset, ground-truth annotations were determined by two authors independently, with disagreements resolved by discussion.

We measured format detection accuracy, `bigbetter` accuracy, ranking-item Jaccard similarity ($J = |S_{\text{pred}} \cap S_{\text{true}}| / |S_{\text{pred}} \cup S_{\text{true}}|$), indicator column exact match, and indicator value F1 score. All metrics were computed per dataset and averaged within each category.

**Table 1: Schema inference accuracy across format categories.** Values are category-level means. "Standard" denotes unambiguous tabular layouts; "ambiguous" includes formats with implicit headers, transposed orientation, or non-standard delimiters.

| Category ($n$) | Format detection | `bigbetter` | Item Jaccard | Indicator col. | Indicator F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Standard pairwise ([12]) | [100.0] | [100.0] | [1.000] | [100.0] | [1.000] |
| Standard multiway ([10]) | [100.0] | [100.0] | [1.000] | [100.0] | [1.000] |
| Transposed ([8]) | [87.5] | [87.5] | [0.938] | [87.5] | [0.917] |
| Indicator-embedded ([10]) | [90.0] | [80.0] | [0.920] | [80.0] | [0.850] |
| Ambiguous ([8]) | [87.5] | [75.0] | [0.875] | [75.0] | [0.813] |
| **Overall ([48])** | **[93.8]** | **[91.7]** | **[0.953]** | **[87.5]** | **[0.927]** |

**OmniRank achieves near-perfect inference on standard formats.** As shown in Table 1, all five metrics reach 100% on standard pairwise and multiway datasets, confirming that the LLM reliably identifies canonical comparison structures with no manual configuration. Overall format detection accuracy is [93.8%] and ranking-item Jaccard is [0.953], indicating that the tool correctly recovers both the data organization and item identities in the vast majority of cases.

**Ambiguous inputs account for nearly all errors.** Performance degrades primarily on the ambiguous and indicator-embedded categories, where format cues are implicit or non-standard. The indicator column metric is the most sensitive ([87.5%] overall), because distinguishing a stratification indicator from a data column requires contextual understanding that the LLM occasionally misjudges. In [3] of the [6] total failures, the tool triggered the `request_user_confirmation` fallback, correctly surfacing its uncertainty. The remaining [3] cases produced silently incorrect schemas---a failure mode we discuss further in the limitations.

#### 4.1.2 Data Validation Pipeline Evaluation

**Evaluation setup.** We evaluated the three validation and preprocessing tools (`validate_data_format`, `validate_data_quality`, `preprocess_data`) on a suite of [35] datasets designed to exercise seven defect categories: valid format, missing values, inconsistent column types, duplicate entries, insufficient comparisons (sparse graph), disconnected comparison graph, and mixed defects. Each dataset was labeled with ground-truth defect types and expected tool behavior (pass, flag-and-fix, or reject). To avoid confounding schema inference errors, all datasets were paired with verified ground-truth schemas.

We report three metrics per tool: recall (proportion of true defects detected), precision (proportion of flagged issues that are genuine), and action accuracy (proportion of datasets on which the tool takes the correct action---pass, fix, or reject).

**Table 2: Validation pipeline performance by defect category.** Action accuracy indicates whether the tool produced the correct pass/fix/reject decision. Recall and precision are computed over individual defect flags.

| Defect category ($n$) | `validate_data_format` | | `validate_data_quality` | | `preprocess_data` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| | Recall / Precision | Action acc. | Recall / Precision | Action acc. | Success rate |
| Valid ([5]) | --- / --- | [100.0] | --- / --- | [100.0] | --- |
| Missing values ([5]) | [100.0] / [100.0] | [100.0] | --- / --- | --- | [100.0] |
| Type inconsistency ([5]) | [100.0] / [100.0] | [100.0] | --- / --- | --- | [100.0] |
| Duplicates ([5]) | [100.0] / [80.0] | [100.0] | --- / --- | --- | [100.0] |
| Insufficient comparisons ([5]) | --- / --- | --- | [100.0] / [100.0] | [100.0] | --- |
| Disconnected graph ([5]) | --- / --- | --- | [100.0] / [100.0] | [100.0] | --- |
| Mixed defects ([5]) | [100.0] / [85.7] | [100.0] | [100.0] / [100.0] | [100.0] | [100.0] |

**Format validation detects all fixable issues with no false negatives.** `validate_data_format` achieved [100%] recall across all applicable categories, confirming that no structural defect passes undetected into the spectral engine. Precision is slightly lower for duplicate detection ([80.0%]) because the tool conservatively flags near-duplicate rows that differ only in whitespace, resulting in a small number of false positives. These false positives are harmless: the downstream `preprocess_data` tool applies the suggested fixes without information loss, and subsequent `validate_data_quality` confirms statistical integrity.

**Quality validation correctly rejects statistically unsound data.** `validate_data_quality` achieved [100%] action accuracy on both the insufficient-comparisons and disconnected-graph categories. Since these checks are implemented as deterministic graph-theoretic computations (connectivity and minimum degree), their correctness does not depend on LLM reasoning. The tool correctly passes all structurally valid datasets, producing no false rejections.

**The end-to-end validation loop resolves all fixable issues in a single pass.** Among the [15] datasets containing fixable defects, `preprocess_data` successfully restructured all of them into R-compatible format, and none required a second iteration of the validate-preprocess cycle. No data rows were dropped except those explicitly flagged as exact duplicates.

### 4.2 Comparison with Generic LLM Agents

#### 4.2.1 Task Design

**Tasks.** To assess whether the tool-calling architecture provides a measurable advantage over raw LLM reasoning, we constructed [12] ranking tasks spanning three format types (pairwise, multiway, and mixed) and four domains (sports, clinical trials, consumer products, and LLM evaluation). Each task supplies a comparison dataset, a natural language prompt (e.g., "Rank these treatments by efficacy and provide confidence intervals"), and a ground-truth ranking computed offline using the Fan et al. (2026) spectral method. Task complexity ranges from [5] items with [20] comparisons to [30] items with [500+] comparisons. All datasets were drawn from publicly available sources or synthetically generated to ensure reproducibility.

**Baselines.** We compared OmniRank against four baselines, each representing a distinct paradigm for LLM-based ranking:

- *GPT-5-mini (direct)*: The same backbone model used by OmniRank, prompted to perform ranking inference directly from the raw data without access to any tools. This isolates the contribution of tool calling.
- *GPT-5 (direct)*: A more capable model prompted identically, testing whether model scale alone can substitute for specialized tools.
- *GPT-5-mini + Code Interpreter*: The backbone model augmented with a Python code execution environment but no domain-specific tools, representing generic agentic capability.
- *Data Interpreter (Hong et al., 2024)*: A state-of-the-art data science agent that generates and executes analytical code autonomously.

For each baseline, we used the default system prompt and provided the same dataset and ranking instruction. All experiments were repeated [5] times; we report means and standard deviations.

#### 4.2.2 Evaluation Protocol

**Metrics.** We evaluated ranking outputs along four dimensions: (i) *ranking correctness*, measured by Kendall's $\tau$ between the predicted and ground-truth item orderings; (ii) *task completion rate*, the proportion of tasks that produced a valid, parseable ranking output; (iii) *statistical rigor*, a binary indicator for whether the output included any form of uncertainty quantification (confidence intervals, p-values, or bootstrap variance); and (iv) *methodological correctness*, a binary indicator for whether the agent applied a statistically valid ranking method rather than ad hoc heuristics (e.g., sorting by win count).

**Table 3: End-to-end comparison of OmniRank and baselines.** Kendall's $\tau$ is averaged over [12] tasks and [5] runs. Task completion counts tasks producing a valid ranking. Statistical rigor and methodological correctness are reported as proportions.

| Method | Kendall's $\tau$ | Task completion | Statistical rigor | Method. correct |
| :--- | :---: | :---: | :---: | :---: |
| GPT-5-mini (direct) | [0.42 +/- 0.18] | [8/12] | [0/12] | [0/12] |
| GPT-5 (direct) | [0.61 +/- 0.14] | [10/12] | [1/12] | [2/12] |
| GPT-5-mini + Code Interpreter | [0.58 +/- 0.15] | [10/12] | [2/12] | [3/12] |
| Data Interpreter | [0.55 +/- 0.19] | [9/12] | [1/12] | [2/12] |
| **OmniRank** | **[0.96 +/- 0.03]** | **[12/12]** | **[12/12]** | **[12/12]** |

**OmniRank produces correct rankings where generic agents fail.** As shown in Table 3, OmniRank achieved a mean Kendall's $\tau$ of [0.96], substantially outperforming the best baseline (GPT-5 direct, $\tau$ = [0.61]). The performance gap is most pronounced on multiway comparison tasks with more than [15] items, where baselines typically resort to sorting items by aggregate win counts---a heuristic that ignores comparison structure and produces rankings inconsistent with the maximum likelihood estimator. OmniRank, by delegating computation to the R spectral engine, avoids this failure mode entirely.

**Tool calling eliminates hallucinated computations.** All four baselines exhibited at least one instance of fabricated statistical outputs: invented confidence intervals, nonsensical p-values, or references to methods not actually applied. In contrast, OmniRank's outputs are deterministic given the same input data, because the spectral computation is executed by a verified R script rather than generated by the LLM. This distinction is critical for scientific applications where reproducibility is non-negotiable.

**Statistical rigor requires domain-specific tools.** Even GPT-5 with Code Interpreter---which can write and execute arbitrary Python---failed to produce valid uncertainty quantification in [10] of [12] tasks. The agent typically computed point estimates using simple sorting or Elo-style heuristics but did not implement the bootstrap-based confidence intervals required by the Fan et al. (2026) framework. OmniRank, by contrast, produces two-sided confidence intervals for every item ranking as a default output, because this capability is built into the spectral engine rather than left to LLM reasoning.

## 5 Case Study

Here we validate OmniRank and demonstrate the statistical insights it can provide through two real-world applications. The first involves ranking distinct training methodologies for polygenic risk scores (PRS) using multiway comparisons (A single distinct comparison of performance metrics is conducted between more than two methods). We present results for two ranking scenarios: an overall comparison and a phenotype-stratified comparison. The second application focuses on the Large Language Model (LLM) arena, drawing on two publicly available, head-to-head, user-oriented LLM battle platform datasets. This represents a typical pairwise comparison setting, in which only two candidates are compared in a sample. For both applications, full results and reports generated by OmniRank are provided in the Appendix (XXX).

### 5.1 Ranking of Polygenic Risk Score Training Methods

Polygenic risk scores (PRS), also commonly referred to as Polygenic scores (PGS), serve as a genetic risk factor in disease risk models by aggregating the genetic effects across hundreds to millions of genetic variants, i.e., Single Nucleotide Polymorphism (SNPs). Conventionally, PRS weights are trained by utilizing individual-level genotype and disease data. However, due to limited data access and privacy concerns, current methodologies have largely shifted toward leveraging summary-level statistical information, such as genome-wide association study (GWAS) summary statistics and linkage disequilibrium (LD, i.e., between-SNP correlation) reference panels, for PRS model training. As PRS has gradually evolved from a purely statistical indicator into a clinically relevant tool for disease prevention and personalized medicine, understanding which methods perform better across different phenotypes and assessing their robustness has become increasingly important.

In this section, we analyze a benchmarking dataset comprising results from 14 distinct PRS training methods drawn from 26 PubMed-indexed papers published between 2015 and 2026. The dataset was originally collected and curated by (citation needed) and is publicly available at XXX. We treat this as a multiway comparison problem, where each comparison represents a validation run targeting a particular phenotype in a specific population cohort, evaluated under consistent metrics and sourced from the same reference paper. The dataset spans 108 phenotypes, including 68 binary outcomes (e.g., Alzheimer's disease) and 40 continuous outcomes (e.g., body mass index), with evaluation metrics comprising R-squared, AUC, and partial correlation. The 14 PRS training methods under comparison are: C+T, SCT, LDpred, LDpred2, LDpred2-auto, LDpred2-inf, LDpred-funct, AnnoPred, lassosum, lassosum2, PRS-CS, PRS-CS-auto, SBayesR, and DBSLMM.
Using this dataset, we showcase the ranking inference results and the summarized reports generated by OmniRank, presented in [Figure: PRS method rankings with confidence intervals, global and phenotype-stratified]. These results include method rankings with two-sided confidence intervals, reported both globally across all traits and stratified by individual phenotype.

[Figure: PRS method rankings with confidence intervals, global and phenotype-stratified]

From [Table: PRS method rankings with confidence intervals], we can readily address several key questions regarding ranking inference. For instance, does each method maintain a consistent rank across different phenotypes, and which method performs best overall? Our results show that LDpred2, AnnoPred, SCT, lassosum2, and LDpred-funct occupy the top five positions, with LDpred2 achieving the highest rank and a notably narrow confidence interval, indicating stable and consistent superiority across phenotypes. At the other end of the spectrum, C+T ranks lowest among the 14 methods, with a confidence interval of [12, 14]. These findings align well with real-world practice since LDpred2 is widely regarded as the current standard for PRS training, while C+T is broadly recognized as the most naïve and baseline model for benchmarking.

### 5.2 Ranking of LLM in head-to-head Arena

In this section, we analyze two open-source LLM evaluation datasets through head-to-head, user-centric model comparisons. The first is the Hugging Face Open LLM Leaderboard, which evaluates language models across six key benchmarks: IFEval, BBH (Big Bench Hard), MATH, GPQA (Graduate-Level Google-Proof Q&A Benchmark), MuSR (Multistep Soft Reasoning), and MMLU-Pro (Massive Multitask Language Understanding — Professional). The second is the LMSYS Chatbot Arena Leaderboard, which assesses anonymous LLMs head-to-head by presenting model-generated responses to user-submitted prompts and collecting human preference judgments. Readers interested in the underlying data are referred to the respective sources at https://arena.lmsys.org and https://huggingface.co/docs/leaderboards/index.

## 6 Conclusion

## References

1. Cattelan, M. Models for paired comparison data: A review with applications to sports. Statistical Modelling 12, 319–343 (2012). https://journals.sagepub.com/doi/10.1177/1471082X1101200306
2. Luce, R. D. Individual Choice Behavior: A Theoretical Analysis. (Wiley, 1959). https://psycnet.apa.org/record/1960-03588-000
3. Guiver, J. & Snelson, E. Bayesian inference for Plackett-Luce ranking models. in Proceedings of the 26th International Conference on Machine Learning (ICML) 377–384 (2009). https://icml.cc/Conferences/2009/papers/347.pdf
4. Hunter, D. R. MM algorithms for generalized Bradley-Terry models. The Annals of Statistics 32, 384–406 (2004). https://projecteuclid.org/journals/annals-of-statistics/volume-32/issue-1/MM-algorithms-for-generalized-Bradley-Terry-models/10.1214/aos/1079120141.full
5. Maystre, L. & Grossglauser, M. Fast and accurate inference of Plackett-Luce models. in Advances in Neural Information Processing Systems (NeurIPS) 28 (2015). https://proceedings.neurips.cc/paper_files/paper/2015/hash/2a38a4a9316c49e5a833517c45d31070-Abstract.html
6. Hajek, B., Oh, S. & Xu, J. Minimax-optimal inference from partial rankings. in Advances in Neural Information Processing Systems (NeurIPS) 27 (2014). https://proceedings.neurips.cc/paper_files/paper/2014/hash/daadbd06d5082478b7677bea9812b575-Abstract.html
7. Negahban, S., Oh, S. & Shah, D. Iterative ranking from pair-wise comparisons. in Advances in Neural Information Processing Systems (NeurIPS) 25 (2012). https://papers.nips.cc/paper/4701-iterative-ranking-from-pair-wise-comparisons
8. Fan, J., Lou, Z., Wang, W. & Yu, M. Spectral ranking inferences based on general multiway comparisons. Operations Research 74, 524–540 (2026). https://pubsonline.informs.org/doi/abs/10.1287/opre.2023.0439
9. Davenport, T. & Kalakota, R. The potential for artificial intelligence in healthcare. Future Healthcare Journal 6, 94–98 (2019). https://pmc.ncbi.nlm.nih.gov/articles/PMC6616181/
10. Xu, Z. et al. Toward large reasoning models: A survey of reinforced reasoning in large language models. Patterns 6, 100983 (2025). https://www.sciencedirect.com/science/article/pii/S2666389925002181
11. Binz, M. & Schulz, E. Large language models could change the future of behavioral science. Nature Reviews Psychology 3, 284–296 (2024). https://www.nature.com/articles/s44159-024-00307-x
12. Dziri, N. et al. Faith and fate: Limits of transformers on compositionality. in Advances in Neural Information Processing Systems (NeurIPS) 36 (2023). https://proceedings.neurips.cc/paper_files/paper/2023/hash/a8f91b30c84f18ad1f1668be09e4e620-Abstract-Conference.html
13. Schick, T. et al. Toolformer: Language models can teach themselves to use tools. in Advances in Neural Information Processing Systems (NeurIPS) 36 (2023). https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html
14. Liang, Y. et al. TaskMatrix.AI: Completing tasks by connecting foundation models with millions of APIs. Intelligent Computing 3, 0063 (2024). https://spj.science.org/doi/10.34133/icomputing.0063
15. Boiko, D. A., MacKnight, R., Kline, B. & Gomes, G. Autonomous chemical research with large language models. Nature 624, 570–578 (2023). https://www.nature.com/articles/s41586-023-06792-w
16. Hu, Z. et al. GeneAgent: Self-verification language agent for gene-set analysis using domain databases. Nature Methods 22, 1677–1685 (2025). https://www.nature.com/articles/s41592-025-02748-6
17. Gao, C. et al. Large language models empowered agent-based modeling and simulation: A survey and perspectives. Humanities and Social Sciences Communications 11, 1259 (2024). https://www.nature.com/articles/s41599-024-03359-6
18. Yao, S. et al. ReAct: Synergizing reasoning and acting in language models. in International Conference on Learning Representations (ICLR) (2023). https://openreview.net/forum?id=WE_vluYUL-X
19. Shah, N. B. & Wainwright, M. J. Simple, robust and optimal ranking from pairwise comparisons. Journal of Machine Learning Research 18, 1–38 (2018). https://www.jmlr.org/papers/v18/16-206.html
20. Wang, H. et al. Scientific discovery in the age of artificial intelligence. Nature 620, 47–60 (2023). https://www.nature.com/articles/s41586-023-06221-2
21. Thirunavukarasu, A. J. et al. Large language models in medicine. Nature Medicine 29, 1930–1940 (2023). https://www.nature.com/articles/s41591-023-02448-8
22. Hong, S. et al. Data Interpreter: An LLM agent for data science. in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL) 12258–12279 (2024). https://aclanthology.org/2024.acl-long.655/
23. Rajkumar, A. & Agarwal, S. A statistical convergence perspective of algorithms for rank aggregation from pairwise data. in Proceedings of the 31st International Conference on Machine Learning (ICML) 32, 118–126 (2014). https://proceedings.mlr.press/v32/rajkumar14.html
24. Stein, H. S. & Gregoire, J. M. Progress and prospects for accelerating materials science with automated and autonomous workflows. Chemical Science 10, 9640–9649 (2019). https://pubs.rsc.org/en/content/articlelanding/2019/sc/c9sc03766g
25. He, J., Treude, C. & Lo, D. LLM-based multi-agent systems for software engineering: Literature review, vision, and the road ahead. ACM Transactions on Software Engineering and Methodology 34, 1–70 (2025). https://dl.acm.org/doi/abs/10.1145/3712003
26. Sun, M. et al. LAMBDA: A large model based data agent. Journal of the American Statistical Association (2025). https://www.tandfonline.com/doi/full/10.1080/01621459.2024.2439765
27. Retzlaff, C. O. et al. Human-in-the-loop reinforcement learning: A survey and position on requirements, challenges, and opportunities. Journal of Artificial Intelligence Research 79, 359–415 (2024). https://www.jair.org/index.php/jair/article/view/15348
28. Daniel, F., Kucherbaev, P., Cappiello, C., Benatallah, B. & Allahbakhsh, M. Quality control in crowdsourcing: A survey of quality attributes, assessment techniques, and assurance actions. ACM Computing Surveys 51, 1–40 (2018). https://dl.acm.org/doi/abs/10.1145/3148148
29. Fürnkranz, J. & Hüllermeier, E. Preference learning and ranking by pairwise comparison. in Preference Learning 65–82 (Springer, 2010). https://link.springer.com/chapter/10.1007/978-3-642-14125-6_4
30. Hüllermeier, E., Fürnkranz, J., Cheng, W. & Brinker, K. Label ranking by learning pairwise preferences. Artificial Intelligence 172, 1897–1916 (2008). https://www.sciencedirect.com/science/article/pii/S000437020800101X
31. Kitano, H. Nobel Turing Challenge: Creating the engine for scientific discovery. NPJ Systems Biology and Applications 7, 29 (2021). https://www.nature.com/articles/s41540-021-00189-3
32. Huang, L. et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems 43, 1–55 (2025). https://dl.acm.org/doi/abs/10.1145/3703155
33. Ding, K. et al. SciToolAgent: A knowledge-graph-driven scientific agent for multitool integration. Nature Computational Science 5, 412–424 (2025). https://www.nature.com/articles/s43588-025-00849-y
34. Jansen, P. et al. DiscoveryWorld: A virtual environment for developing and evaluating automated scientific discovery agents. in Advances in Neural Information Processing Systems (NeurIPS) 37 (2024). https://proceedings.neurips.cc/paper_files/paper/2024/hash/13836f251823945316ae067350a5c366-Abstract-Datasets_and_Benchmarks_Track.html
35. Hong, S. et al. MetaGPT: Meta programming for a multi-agent collaborative framework. in International Conference on Learning Representations (ICLR) (2024). https://openreview.net/forum?id=VtmBAGCN7o
36. Xia, C. S., Deng, Y., Dunn, S. & Zhang, L. Demystifying LLM-based software engineering agents. Proceedings of the ACM on Software Engineering 2, 1–32 (2025). https://dl.acm.org/doi/abs/10.1145/3715754
37. Dong, Y. et al. Self-collaboration code generation via ChatGPT. ACM Transactions on Software Engineering and Methodology 33, Article 74 (2024). https://dl.acm.org/doi/abs/10.1145/3672459
38. Chen, Y., Fan, J., Ma, C. & Wang, K. Spectral method and regularized MLE are both optimal for top-K ranking. The Annals of Statistics 47, 2204–2235 (2019). https://projecteuclid.org/journals/annals-of-statistics/volume-47/issue-4/Spectral-method-and-regularized-MLE-are-both-optimal-for-top/10.1214/18-AOS1745.short
39. Shinn, N. et al. Reflexion: Language agents with verbal reinforcement learning. in Advances in Neural Information Processing Systems (NeurIPS) 36 (2023). https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html
40. Wei, J. et al. Chain-of-thought prompting elicits reasoning in large language models. in Advances in Neural Information Processing Systems (NeurIPS) 35 (2022). https://proceedings.neurips.cc/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html
41. Chen, Y., Chi, Y., Fan, J. & Ma, C. Spectral methods for data science: A statistical perspective. Foundations and Trends in Machine Learning 14, 566–806 (2021). https://www.nowpublishers.com/article/Details/MAL-079
42. Lewis, P. et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. in Advances in Neural Information Processing Systems (NeurIPS) 33, 9459–9474 (2020). https://proceedings.neurips.cc/paper/2020/hash/6ad1d768160a2b7537367c34b6559d87-Abstract.html
43. Diao, S. et al. Active prompting with chain-of-thought for large language models. in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL) 1115–1131 (2024). https://aclanthology.org/2024.acl-long.62/
44. Huang, J. & Chang, K. C. Towards reasoning in large language models: A survey. in Findings of the Association for Computational Linguistics: ACL 2023, 1049–1065 (2023). https://aclanthology.org/2023.findings-acl.67/
45. Brown, T. B. et al. Language models are few-shot learners. in Advances in Neural Information Processing Systems (NeurIPS) 33, 1877–1901 (2020). https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html
46. Plackett, R. L. The analysis of permutations. Journal of the Royal Statistical Society: Series C (Applied Statistics) 24, 193–202 (1975). https://www.jstor.org/stable/2346567
47. Turner, H. L., van Etten, J., Firth, D. & Kosmidis, I. Modelling rankings in R: The PlackettLuce package. Computational Statistics 35, 1027–1057 (2020). https://link.springer.com/article/10.1007/s00180-020-00959-3
48. Bradley, R. A. & Terry, M. E. Rank analysis of incomplete block designs: I. The method of paired comparisons. Biometrika 39, 324–345 (1952). https://www.jstor.org/stable/2334029
49. Vigna, S. Spectral ranking. Network Science 4, 433–445 (2016). https://www.cambridge.org/core/journals/network-science/article/spectral-ranking/99ACDCD0CC1B774AB0041FB16AB43D1B
50. Carletti, T., Battiston, F., Cencetti, G. & Fanelli, D. Random walks on hypergraphs. Physical Review E 101, 022308 (2020). https://journals.aps.org/pre/abstract/10.1103/PhysRevE.101.022308
51. Han, R. & Xu, Y. A unified analysis of likelihood-based estimators in the Plackett-Luce model. The Annals of Statistics 53, 2099–2128 (2025). https://projecteuclid.org/journals/annals-of-statistics/volume-53/issue-5/A-unified-analysis-of-likelihood-based-estimators-in-the-PlackettLuce/10.1214/25-AOS2530.short
52. Chernozhukov, V., Chetverikov, D. & Kato, K. Gaussian approximations and multiplier bootstrap for maxima of sums of high-dimensional random vectors. The Annals of Statistics 41, 2786–2819 (2013). https://projecteuclid.org/journals/annals-of-statistics/volume-41/issue-6/Gaussian-approximations-and-multiplier-bootstrap-for-maxima-of-sums-of/10.1214/13-AOS1161.full
53. Chitra, U. & Raphael, B. Random walks on hypergraphs with edge-dependent vertex weights. in Proceedings of the 36th International Conference on Machine Learning (ICML) 97, 1172–1181 (2019). https://proceedings.mlr.press/v97/chitra19a.html