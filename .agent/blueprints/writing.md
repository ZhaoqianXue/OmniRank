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

This section presents the methodological framework of OmniRank, an agentic system that bridges large language model reasoning with rigorous spectral ranking inference. We first provide an architectural overview (Section 3.1), then detail the three core components: the Data Agent for intelligent data preprocessing (Section 3.2), the Engine Orchestrator for adaptive statistical computation (Section 3.3), and the Analyst Agent for result interpretation and user interaction (Section 3.4). We subsequently describe the prompt engineering strategies (Section 3.5), user interface design (Section 3.6), and conclude with the mathematical foundations of the spectral ranking engine (Section 3.7).

### 3.1 Overview

OmniRank adopts a modular multi-agent architecture designed to separate semantic understanding from mathematical computation, addressing a fundamental limitation of current LLMs: their propensity for hallucination in arithmetic and algorithmic tasks [32]. This architectural principle, termed "decoupled reasoning," has proven effective in recent scientific agent systems [33, 34] and forms the theoretical basis for our design.

The system comprises three functionally distinct components operating in a coordinated pipeline:

1. **Data Agent**: An LLM-powered component responsible for data format recognition, validation, and semantic schema inference. The agent interprets user-uploaded datasets and extracts structural metadata necessary for spectral computation.

2. **Engine Orchestrator**: A deterministic controller that manages the interface between semantic understanding and statistical computation. It executes the spectral ranking engine with robust process management and output parsing.

3. **Analyst Agent**: An LLM-powered component that synthesizes computational results into interpretable outputs, generates visualizations, and supports interactive question-answering through domain knowledge integration.

Figure 1 illustrates the system architecture and information flow between components. The workflow proceeds through four phases: data ingestion and schema inference, interactive parameter configuration, spectral computation with adaptive refinement, and result synthesis with user interaction. Algorithm 1 formalizes this process.

**Figure 1: OmniRank System Architecture.** The system comprises three core components: the Data Agent (LLM-powered) for semantic understanding, the Engine Orchestrator (deterministic) for statistical computation, and the Analyst Agent (LLM-powered) for result interpretation. Solid arrows indicate data flow; dashed arrows indicate feedback loops for error handling.

**Algorithm 1** OmniRank Workflow

**Input:** Dataset $\mathcal{D}$ uploaded by user; maximum retry attempts $T$
**Output:** Ranking results $\mathcal{R}$ with confidence intervals; analysis report

*Phase 1: Data Processing and Schema Inference*
1: $\mathcal{S} \leftarrow \texttt{DataAgent.InferSchema}(\mathcal{D})$ $\triangleright$ Infer format, items, indicators, preference direction
2: $\mathcal{V} \leftarrow \texttt{DataAgent.Validate}(\mathcal{D}, \mathcal{S})$ $\triangleright$ Check connectivity, sparsity, data integrity
3: **if** $\mathcal{V}$.status $=$ INVALID **then**
4:     **return** $\texttt{DataAgent.GenerateFeedback}(\mathcal{V})$
5: **end if**

*Phase 2: Interactive Configuration*
6: $\mathcal{P} \leftarrow \texttt{Orchestrator.Configure}(\mathcal{S}, \texttt{UserInput})$ $\triangleright$ User verifies/adjusts parameters
7: $\texttt{Memory.UpdateState}(\mathcal{S}, \mathcal{P})$

*Phase 3: Spectral Computation*
8: $\mathcal{R} \leftarrow \texttt{Orchestrator.Execute}(\mathcal{D}, \mathcal{P})$ $\triangleright$ Spectral estimation

*Phase 4: Error Handling and Output Generation*
14: $n \leftarrow 0$
15: **while** $\mathcal{R}.\text{status} = \text{ERROR}$ **and** $n < T$ **do**
16:     $n \leftarrow n + 1$
17:     $(\text{diagnosis}, \text{suggestions}) \leftarrow \texttt{AnalystAgent.Diagnose}(\mathcal{R}.\text{error})$
18:     $\mathcal{P} \leftarrow \texttt{Orchestrator.AdjustParams}(\mathcal{P}, \text{suggestions})$
19:     $\mathcal{R} \leftarrow \texttt{Orchestrator.ReExecute}(\mathcal{D}, \mathcal{P})$
20: **end while**
21: **return** $\texttt{AnalystAgent.GenerateReport}(\mathcal{R})$

This architecture instantiates the "programmer-inspector" paradigm that has demonstrated superior reliability compared to end-to-end LLM approaches [35]. By delegating precise numerical computation to a verified statistical engine while leveraging LLM capabilities for natural language understanding and explanation, OmniRank achieves both accessibility and mathematical rigor.

### 3.2 Data Agent

The Data Agent serves as the intelligent interface between raw user data and the spectral computation engine. Unlike generic data analysis agents that rely solely on LLM code generation [36], our Data Agent employs a hybrid approach: LLM-based semantic reasoning for schema inference combined with deterministic validation rules grounded in spectral ranking theory. This design reflects recent findings that specialized domain knowledge significantly improves agent performance on knowledge-intensive tasks [37].

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

This tiered validation approach, illustrated in Figure 2, ensures that users receive actionable feedback about data limitations while permitting valid exploratory analyses on imperfect datasets.

**Figure 2: Data Agent Validation Workflow.** The flowchart depicts the hierarchical validation process: critical errors block execution, warnings inform users of theoretical limitations, and valid data proceeds to schema inference.

#### 3.2.2 Semantic Schema Inference

Beyond structural validation, the Data Agent infers the semantic meaning of data components to enable flexible downstream analysis. This capability distinguishes OmniRank from traditional statistical software that requires explicit parameter specification.

**Preference Direction Inference.** The agent determines whether higher metric values indicate superior performance (e.g., accuracy, win rate) or inferior performance (e.g., latency, error rate). This inference combines lexical analysis of column names with distributional properties of the data. For instance, columns containing terms such as "accuracy" or "score" suggest a higher-is-better interpretation, while "time" or "error" suggest the opposite.

**Entity and Indicator Extraction.** The agent identifies:
- *Ranking Items*: The entities to be ranked (e.g., model names, player identifiers).
- *Ranking Indicators*: Categorical dimensions that partition comparisons into semantically meaningful subgroups (e.g., task categories, evaluation conditions).

When multiple potential indicator columns exist, the agent selects at most one to maintain analytical focus, prioritizing columns with moderate cardinality and clear semantic interpretation.

This metadata extraction enables the Engine Orchestrator to present users with intuitive configuration options, allowing customized analysis without requiring statistical expertise.

### 3.3 Engine Orchestrator

The Engine Orchestrator is a deterministic system component that manages the transition from semantic schema to statistical computation. Unlike the LLM-powered agents, the orchestrator implements fixed algorithmic logic to ensure reproducibility and numerical accuracy. Its design reflects the "tool-use" paradigm in agentic AI, where LLMs serve as cognitive controllers while delegating precise computations to specialized tools [39, 40].

#### 3.3.1 Interactive Configuration Management

The orchestrator exposes inferred parameters through an interactive configuration interface, enabling users to verify and adjust settings before computation. Configurable parameters include:

- **Preference Direction**: Users confirm or override the inferred interpretation of metric values.
- **Item Selection**: Users may restrict analysis to a subset of items.
- **Indicator Selection**: Users select which indicator values to include in the analysis.
- **Statistical Parameters**: Advanced users may configure bootstrap iterations (default: 2000) and random seed (default: 42) for reproducibility.

This human-in-the-loop design addresses a known limitation of fully automated agent systems: misalignment between inferred parameters and user intent [35]. By requiring explicit confirmation, OmniRank ensures that final analyses reflect user requirements.

#### 3.3.2 Spectral Estimation Execution

The orchestrator invokes the spectral engine with uniform weighting $f(A_l) = |A_l|$ to obtain consistent preference score estimates $\hat{\boldsymbol{\theta}}$. This spectral estimator provides stable estimates across diverse data conditions and achieves minimax optimal rates [8].

The execution workflow consists of:

1. **Parameter Preparation**: Constructing the R script command with validated parameters (data path, bigbetter direction, bootstrap iterations, random seed).

2. **Engine Invocation**: Executing `spectral_ranking_step1.R` in an isolated subprocess with timeout protection.

3. **Output Parsing**: Processing the JSON output to extract preference scores, rankings, and confidence intervals.

4. **Trace Logging**: Recording execution parameters and results in session memory for potential error diagnosis.

This deterministic workflow ensures reliable and reproducible ranking computations.

### 3.4 Analyst Agent

The Analyst Agent is responsible for transforming computational outputs into interpretable results and supporting ongoing user interaction. This component addresses a critical gap in statistical software: the translation of numerical outputs into actionable insights accessible to domain experts without statistical training.

#### 3.4.1 Report and Visualization Generation

Upon receiving ranking results from the Engine Orchestrator, the Analyst Agent synthesizes comprehensive analysis reports through LLM-based natural language generation. Reports include:

- **Executive Summary**: Key findings highlighting top-ranked items and notable patterns.
- **Detailed Rankings**: Tabular presentation of ranks, preference scores, and confidence intervals with statistical significance indicators.
- **Methodology Notes**: Explanation of the spectral approach and validation outcomes.
- **Domain-Specific Insights**: Contextual interpretation tailored to the data domain, leveraging the semantic schema inferred by the Data Agent.

The agent generates a complementary suite of visualizations:

1. *Rank Plots*: Forest plots displaying point estimates with confidence interval error bars, enabling visual assessment of ranking uncertainty [41].
2. *Comparison Heatmaps*: Matrix visualizations of pairwise win rates revealing competitive structure among items.
3. *Score Distributions*: Density plots of estimated preference parameters $\hat{\theta}_i$ illustrating the separation between items.

These outputs are rendered in both interactive web formats and exportable static formats (PDF, PNG) suitable for publication.

#### 3.4.2 Interactive Question-Answering

The Analyst Agent supports follow-up queries through a conversational interface, enabling users to explore results without restarting the analysis. This capability is implemented through a session memory architecture comprising three components:

- **Data State**: Current schema, validation results, and configuration parameters.
- **Execution Trace**: Log of computation invocations and intermediate results for error diagnosis.
- **Conversation Context**: History of user queries and agent responses enabling contextual follow-up.

The agent interprets queries by combining session context with domain knowledge embedded in its system prompt. For example, when a user asks "Is model A significantly better than model B?", the agent retrieves the relevant confidence intervals and applies the non-overlapping confidence interval heuristic to provide a statistically grounded response.

This retrieval-augmented generation approach [42] ensures responses are grounded in computed results rather than hallucinated, addressing a known failure mode of vanilla LLM applications to quantitative domains [32].

### 3.5 Prompt Engineering

We adopt structured system prompts following established practices in LLM agent design [43, 44]. Each agent's prompt comprises three layers: role specification, operational constraints, and domain knowledge.

**Role Specification.** Defines the agent's identity and primary responsibilities. For example, the Data Agent is instructed to act as a "statistical data analyst specializing in comparison data formats and ranking analysis."

**Operational Constraints.** Specifies output formats, error handling procedures, and interaction protocols. These constraints ensure consistent behavior across diverse inputs and enable reliable parsing of agent outputs.

**Knowledge Layer.** Embeds domain expertise directly into the prompt, enabling expert-level reasoning without requiring fine-tuning. The Data Agent's knowledge layer includes format recognition rules and validation thresholds; the Analyst Agent's knowledge layer includes spectral ranking theory concepts such as confidence interval interpretation and ranking diagnostics.

This knowledge integration approach, illustrated in Figures 4 and 5, follows the in-context learning paradigm [45] that has proven effective for knowledge-intensive tasks without model modification.

**Figure 4: Data Agent System Prompt Structure.** The prompt comprises role specification, format recognition rules, validation thresholds derived from spectral ranking theory, and output format constraints.

**Figure 5: Analyst Agent System Prompt Structure.** The prompt includes role specification, spectral ranking domain knowledge (confidence intervals, ranking thresholds), and report generation guidelines.

### 3.6 User Interface

OmniRank provides a web-based conversational interface designed for accessibility across user expertise levels. The interface guides users through a three-stage workflow:

**Stage 1: Data Upload and Analysis.** Users upload comparison data in standard formats (CSV, Excel). The Data Agent processes the upload, displaying format recognition results, validation outcomes, and inferred schema parameters in an organized panel.

**Stage 2: Interactive Configuration.** The interface presents inferred settings in a visual control panel where users can confirm or modify preference direction, select items and indicator values, and configure advanced statistical parameters. This stage ensures alignment between system inference and user intent before computation proceeds.

**Stage 3: Results and Exploration.** Upon computation completion, the interface displays ranking results with interactive visualizations and a natural language summary. A chat panel enables follow-up queries such as "Which items have statistically indistinguishable rankings?" or "What would change if we excluded item X?"

Figure 6 presents interface screenshots illustrating each stage.

**Figure 6: OmniRank User Interface.** Panel (a) shows the data upload and schema inference display; panel (b) shows the interactive configuration panel; panel (c) shows the results dashboard with visualizations and chat interface.

### 3.7 Spectral Ranking Inference Engine

The mathematical foundation of OmniRank rests on spectral methods for ranking inference from comparison data. This section provides the theoretical basis for the computation implemented in the Engine Orchestrator.

#### 3.7.1 Problem Formulation

Consider $n$ items to be ranked based on comparison outcomes. We model preferences through the Plackett-Luce framework [46, 47], parameterizing item quality by $\boldsymbol{\theta}^* = (\theta_1^*, \ldots, \theta_n^*)^\top$ such that for any choice set $A \subseteq [n]$ and item $i \in A$:

$$P(i \text{ wins among } A) = \frac{e^{\theta_i^*}}{\sum_{k \in A} e^{\theta_k^*}}$$

This model encompasses the Bradley-Terry-Luce model for pairwise comparisons as a special case when $|A| = 2$ [48].

The observed data consist of $L$ comparisons $\{(c_l, A_l)\}_{l=1}^L$, where $A_l$ is the choice set for comparison $l$ and $c_l \in A_l$ denotes the winner. This formulation accommodates heterogeneous comparison structures where choice sets may vary in size, a common characteristic of real-world ranking data [8].

#### 3.7.2 Spectral Estimation via Random Walks

The spectral approach constructs a Markov chain over items whose stationary distribution reflects latent preferences [49, 50]. Define the transition matrix $\mathbf{P}$ with entries:

$$P_{ij} = \frac{1}{d_i} \sum_{l \in W_j \cap L_i} \frac{1}{f(A_l)}$$

where $W_j = \{l : j \in A_l, c_l = j\}$ indexes comparisons won by $j$, $L_i = \{l : i \in A_l, c_l \neq i\}$ indexes comparisons lost by $i$, $d_i = \sum_{j \neq i} P_{ij}$ is a normalizing constant, and $f(A_l)$ is a weighting function.

The stationary distribution $\hat{\boldsymbol{\pi}}$ of this chain, obtained as the leading eigenvector of $\mathbf{P}^\top$, serves as the spectral score vector. Preference parameters are recovered via the log-transformation:

$$\tilde{\theta}_i = \log \hat{\pi}_i - \frac{1}{n} \sum_{k=1}^{n} \log \hat{\pi}_k$$

yielding centered estimates comparable across different analyses.

#### 3.7.3 Uncertainty Quantification

Confidence intervals for ranking parameters are constructed using the Gaussian multiplier bootstrap [52, 53]. Let $\{e_l\}_{l=1}^L$ be i.i.d. standard normal random variables independent of the data. The bootstrap distribution:

$$\hat{\boldsymbol{\theta}}^* - \hat{\boldsymbol{\theta}} \mid \text{data}$$

approximates the sampling distribution of $\hat{\boldsymbol{\theta}} - \boldsymbol{\theta}^*$, enabling construction of confidence intervals without parametric assumptions on the comparison process [8].

For rank inference, we employ the bootstrap to assess whether observed rank differences are statistically significant, providing users with rigorous uncertainty quantification for ranking conclusions.

## 4 Experiments

In this section, we evaluate OmniRank's performance through three experimental dimensions: (1) verifying the computational fidelity of the spectral ranking engine, (2) evaluating the intelligent capabilities of OmniRank's agent components, and (3) demonstrating OmniRank's advantage over generic data analysis agents.

### 4.1 Computational Fidelity Validation

**Purpose**: Verify that OmniRank's spectral ranking engine produces mathematically correct results. We compare agent-mediated execution to direct R package calls to confirm that the delegation pipeline introduces no precision loss.

**What to include**:
- **Experimental Setup**: 
  - Synthetic datasets generated from Plackett-Luce model
  - Parameter ranges: $n \in \{50, 100, 200\}$ items, varying comparison counts $M$
  - Baseline: Standard R implementation of spectral ranking (Yu et al. [8])
- **Metrics**: 
  - Spearman correlation coefficient ($\rho$) between true and estimated preference scores
  - Ranking Mean Squared Error (RMSE)
- **Results Table**: 
  - OmniRank vs R package results side-by-side
  - Should show numerical identity (e.g., $\rho > 0.98$ for $M > n \log n$)
- **Discussion**: 
  - Confirm that LLM-driven "Instruction Following to Computation" delegation does not introduce numerical artifacts
  - Note that this is a necessary but not sufficient validation (the real value is in agentic capabilities)

### 4.2 OmniRank Agent Capability Evaluation

This section evaluates the intelligent capabilities of OmniRank's agent components. Unlike the computational fidelity validation in Section 4.1, which verifies numerical correctness, these experiments assess the agents' ability to perform semantic understanding---capabilities that distinguish OmniRank from simple wrapper systems. We designed a comprehensive test suite covering two core functions: data format recognition (Section 4.2.1) and semantic schema inference (Section 4.2.2).

#### 4.2.1 Format Recognition and Validation

**Experimental Setup.** The Data Agent must automatically identify the structure of uploaded data and assess its suitability for spectral ranking analysis. We constructed a test suite of 41 synthetic datasets spanning five difficulty categories: (i) *Standard* formats (9 datasets) with clean, unambiguous structures; (ii) *Ambiguous* cases (8 datasets) where data characteristics could lead to misclassification (e.g., sparse pointwise data resembling pairwise comparisons, binary values identical to win/loss encodings); (iii) *Transposed* structures (4 datasets) with rows and columns swapped from expected orientations; (iv) *Invalid* data (8 datasets) that should trigger rejection (single-column files, all-text content); and (v) *Real-world* formats (12 datasets) based on common internet data sources such as tennis match records and product ratings.

Each dataset was labeled with ground truth for three classification targets: data format (pointwise, pairwise, multiway, or invalid), engine compatibility (compatible or incompatible), and required standardization action (none, standardize, or reject).

**Results.** Table 1 presents the format detection performance. The Data Agent achieved 92.7% overall accuracy across the 41 test datasets. Performance varied across categories: the agent achieved perfect accuracy on Standard datasets (100%) and near-perfect on Transposed structures (100%), demonstrating reliable handling of both clean inputs and structurally challenging orientations. On Ambiguous cases, accuracy reached 87.5%, indicating robust discrimination between superficially similar formats.

**Table 1: Format Detection Performance of the Data Agent**

| Format | Precision | Recall | F1 Score | Support |
|--------|-----------|--------|----------|---------|
| Pointwise | 0.950 | 1.000 | 0.974 | 19 |
| Pairwise | 0.900 | 1.000 | 0.947 | 9 |
| Multiway | 1.000 | 0.889 | 0.941 | 9 |
| Invalid | 0.800 | 0.750 | 0.774 | 4 |
| **Overall Accuracy** | | | **0.927** | 41 |

The agent demonstrated strong performance on Real-world datasets (91.7% accuracy), reflecting its capacity for semantic understanding---correctly interpreting winner/loser column pairs in tennis match data, recognizing benchmark score matrices despite irregular column naming conventions, and handling varied real-world formatting conventions.

**Table 2: Category-wise Format Detection Accuracy**

| Category | Accuracy |
|----------|----------|
| Standard | 100.0% |
| Ambiguous | 87.5% |
| Transposed | 100.0% |
| Invalid | 75.0% |
| Real-world | 91.7% |

Error analysis revealed two systematic failure modes. First, the agent occasionally misclassified invalid data (75% accuracy), with some single-column files incorrectly identified as valid pointwise format. Second, ambiguous two-item multiway rankings (values 1 and 2 only) were occasionally confused with pairwise comparisons. These errors suggest opportunities for prompt engineering refinements in future iterations.

#### 4.2.2 Semantic Schema Inference

**Experimental Setup.** Beyond format recognition, the Data Agent must infer semantic properties of the data that are essential for correct ranking interpretation. We evaluated four schema inference tasks: (i) *BigBetter direction*---determining whether higher or lower metric values indicate superior performance (e.g., accuracy vs. error rate); (ii) *Ranking items identification*---correctly identifying which columns represent the items to be ranked; (iii) *Indicator column detection*---recognizing categorical columns that define subgroups for stratified analysis; and (iv) *Indicator values extraction*---correctly enumerating the unique values within indicator columns.

We constructed 44 test datasets across six categories designed to probe different aspects of semantic understanding. The *BigBetter* categories (24 datasets) included metrics where higher is better (accuracy, score, win rate), lower is better (error, cost, latency, loss), and semantically ambiguous cases (algorithm names, model identifiers). The *Indicator* categories (12 datasets) tested detection of explicit categorical columns versus datasets with no true indicator. The *Items Complex* category (8 datasets) featured challenging item identification scenarios including descriptive column names, mixed naming conventions, and multi-word identifiers.

**Results.** Table 3 summarizes the schema inference performance. The Data Agent achieved perfect accuracy (100%) across all four inference tasks and all six dataset categories. This result demonstrates that carefully engineered prompts with domain-specific knowledge can enable reliable semantic understanding for structured data analysis tasks.

**Table 3: Semantic Schema Inference Performance**

| Category | BigBetter Acc. | Items Jaccard | Indicator Acc. | Values F1 |
|----------|----------------|---------------|----------------|-----------|
| BigBetter High | 1.000 | 1.000 | 1.000 | 1.000 |
| BigBetter Low | 1.000 | 1.000 | 1.000 | 1.000 |
| BigBetter Ambiguous | 1.000 | 1.000 | 1.000 | 1.000 |
| Indicator Clear | 1.000 | 1.000 | 1.000 | 1.000 |
| Indicator None | 1.000 | 1.000 | 1.000 | 1.000 |
| Items Complex | 1.000 | 1.000 | 1.000 | 1.000 |
| **Overall** | **1.000** | **1.000** | **1.000** | **1.000** |

The perfect performance on BigBetter inference is particularly notable because incorrect direction inference would invert all rankings---a catastrophic failure mode. The agent's success reflects explicit prompt engineering that instructs the model to reason about metric semantics: "For metrics like 'accuracy', 'score', 'win_rate', higher values indicate better performance. For metrics like 'error', 'loss', 'latency', lower values are better."

### 4.3 Comparison with Generic Agents

This section demonstrates OmniRank's advantage over generic data analysis agents on knowledge-intensive ranking tasks that require specialized spectral ranking expertise.

#### 4.3.1 Knowledge-Intensive Task Performance

**Purpose**: Show that OmniRank's specialized architecture outperforms generic agents on tasks requiring spectral ranking theory, hypergraph construction, and optimal weighting schemes.

**What to include**:
- **Knowledge-Intensive Tasks**:
  - **Task 1**: Spectral ranking with hypergraph Laplacian construction
  - **Task 2**: Bootstrap confidence interval computation and interpretation
  - **Task 3**: User Q&A requiring spectral ranking knowledge (e.g., "Is model A significantly better than model B?" requires confidence interval interpretation)
  - Each task requires precise handling of spectral ranking theory beyond simple Bradley-Terry models
- **Case Study** (optional but recommended):
  - Detailed example where generic agents fail but OmniRank succeeds
  - Demonstrate specific failure modes (e.g., incorrect hypergraph construction, missing bootstrap CI logic)
  - Highlight how specialized Spectral Calculation Engine enables correct results

#### 4.3.2 Scoring System & Baseline Comparison

**Purpose**: Provide quantitative comparison between OmniRank and generic data analysis agents.

**What to include**:
- **Scoring System Definition** (similar to LAMBDA Table 8):
  - **1.0**: Both code generation and execution successful, correct statistical decisions
  - **0.8**: Code successful but execution error due to environment/configuration issues
  - **0.5**: Code error but execution successful (partial functionality)
  - **0.0**: Code error and execution error, or exceeded runtime limit
- **Comparison Baselines**:
  - GPT-4 Advanced Data Analysis (OpenAI)
  - Data Interpreter [22]
  - Other relevant general-purpose data agents if applicable
- **Results Table** (similar to LAMBDA Table 8):
  - Score comparison across agents for each task
  - Failure reasons annotated (code error, execution error, timeout, etc.)
  - Should show OmniRank consistently achieving 1.0 while generic agents score < 0.3
- **Discussion**:
  - Explain why generic agents struggle (lack of spectral ranking knowledge, no specialized engine)
  - Emphasize OmniRank's unique value: combining LLM reasoning with rigorous mathematical backend

## 5 Case Study

### 5.1

### 5.2

### 5.3

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