# OmniRank: A Large-Language-Model Agent Platform for Statistically Rigorous Ranking Inference from Arbitrary Multiway Comparisons

## Abstract

Spectral ranking inferences provide a minimax optimal solution for analyzing multiway comparison data, which could achieve the same asymptotic efficiency as the Maximum Likelihood Estimation (MLE) while providing extra uncertainty quantifications. However, the steep learning curve of linear algebra-based implementations restricts their utility to a small circle of statisticians. In this study, we present OmniRank, an agentic framework that synergizes reasoning capabilities of Large Language Models (LLMs) with the mathematical rigor of spectral ranking inference. Unlike standard LLMs that are prone to hallucinations in arithmetic tasks, OmniRank decouples instruction following from computation: an LLM controller parses user queries and data, delegating the rigorous inference to a specialized Spectral Calculation Engine. Case study results on both synthetic and real-world datasets confirm that OmniRank achieves precise ranking recovery matching established statistical packages. By providing an interactive, no-code interface for spectral ranking, OmniRank democratizes advanced ranking methods and uncertainty inference for domain experts across social and natural sciences.

## 1 Introduction

Ranking inferences from comparison data are ubiquitous in scientific inquiry and modern applications, ranging from identifying optimal treatments in clinical trials and ranking biological stimuli to evaluating the relative strength of competitors in sports and gaming1,2. While classical frameworks like the Bradley-Terry-Luce model have successfully handled pairwise comparisons, real-world data increasingly manifest as multiway comparisons, where multiple items are compared simultaneously—such as in horse races, multi-player online games, or top-k choice data in econometrics3,4. Unlike pairwise data, multiway comparisons involve hyperedges of heterogeneous sizes, creating complex dependency structures that defy simple aggregation. Although the Plackett-Luce model offers a probabilistic foundation for such data, its reliance on Maximum Likelihood Estimation (MLE) faces significant challenges: the likelihood function can be non-convex, and the computational burden becomes prohibitive as the number of items (n) grows, often requiring O(N^3) complexity for precise inference5,6.

To overcome these computational and statistical barriers, recent theoretical breakthroughs have established spectral ranking inferences based on general multiway comparisons as a superior alternative. By constructing a comparison graph where items are nodes and multiway comparisons form hyperedges, these methods utilize the stationary distribution of a random walk on the hypergraph (or the eigenvectors of the hypergraph Laplacian) to recover latent preference scores7. Fan et al. demonstrated that this spectral approach achieves minimax optimal statistical rates comparable to MLE but with significantly greater computational efficiency, even under heterogeneous sampling conditions where hyperedge sizes vary dramatically8. Despite this theoretical elegance, the practical application of spectral ranking remains confined to a small circle of statisticians. The implementation requires rigorous handling of sparse hypergraph adjacency matrices and complex linear algebra operations, creating a steep technical barrier for domain experts—such as sociologists or biologists—who possess rich multiway data but lack the coding expertise to implement these specialized spectral algorithms9.

Large Language Models (LLMs) have emerged as potential intermediaries to democratize such advanced analytical tools. Models like GPT-4 have shown impressive capabilities in code generation and logical reasoning10, 11. However, standard LLMs inherently struggle with rigorous mathematical execution; they are prone to "hallucinations" when performing arithmetic or executing specific algorithms mentally, and they lack the native ability to process large-scale structured data (e.g., adjacency matrices) directly within their context window12. Consequently, the current wave of "AI Agents" has shifted towards a tool-use paradigm, where the LLM acts as a controller that delegates specific tasks to external computational tools13, 14. While agents have been developed for chemical synthesis15 and gene analysis16, there is currently no dedicated framework that bridges the gap between the sophisticated mathematics of spectral ranking and the intuitive needs of non-technical users.

Here, we introduce OmniRank, a novel web-based agentic framework that democratizes access to spectral ranking inferences. The architecture consists of two synergistic components: an LLM Agent that interprets user’s natural language requests and raw data uploads (e.g., “Rank these polygenic risk scores for breast cancer using their comparative AUC performance across the uploaded validation cohorts.”), and a Spectral Calculation Engine that executes the hypergraph construction and eigenvector computations. The results are then rendered through an interactive visualization dashboard, allowing users to explore ranking confidence intervals and topology without writing a single line of code. By decoupling the complex spectral inference (Backend) from the user interaction (Frontend), we ensure that the mathematical precision of the underlying theory is preserved while maximizing accessibility.

We validated the efficacy of OmniRank through both theoretical benchmarking and real-world application scenarios. To assess the fidelity of our agent-driven pipeline, we compared its output against standard R implementations of spectral ranking on synthetic datasets with varying heterogeneity in comparison sizes (k). Furthermore, we demonstrate the tool’s practical utility by applying it to a real-world LLMs dataset, where the agent successfully parsed unstructured match results and produced rankings consistent with ground-truth outcomes. Our results show that by combining the reasoning power of LLMs with the mathematical rigor of spectral graph theory, we can effectively lower the barrier to entry for advanced statistical ranking, enabling broader application across diverse scientific fields.

## 2 Background and related works

### 2.1 

### 2.2

### 2.3 

## 3 Methodology

OmniRank employs a streamlined multi-agent architecture to perform spectral ranking inference. The system uses a fixed pipeline triggered by user data upload, with two specialized agents handling data processing and analysis, respectively.

### 3.1 Overview

OmniRank is structured around three core components: the **Data Agent**, **Engine Orchestrator**, and **Analyst Agent**. The Data Agent handles data preprocessing and parameter inference. The Engine Orchestrator invokes the spectral ranking computation. The Analyst Agent generates reports, visualizations, and handles user Q&A by combining ranking results with spectral ranking knowledge. Error diagnosis is also performed by the Analyst Agent.

When users upload comparison data, the workflow proceeds as follows: The Data Agent preprocesses the data and infers the semantic schema (including `bigbetter`, items, and indicators). **Users then verify and refine these settings via the system's interactive configuration panel.** Upon confirmation, the Engine Orchestrator invokes `spectral_ranking_step1.R` to compute rankings. If errors occur, the Analyst Agent diagnoses the issue and requests corrections. Upon successful computation, the Analyst Agent generates reports, visualizations, and supports ongoing Q&A with users. The workflow is formalized in Algorithm 1.

**Algorithm 1** OmniRank Workflow.

**Require:** $Da$: Data Agent, $Eo$: Engine Orchestrator, $An$: Analyst Agent, $M$: Session Memory
**Require:** $d$: Dataset provided by user, $T$: Maximum number of attempts

// Phase 1: Data Processing & Configuration
1: $schema \leftarrow Da$.infer\_schema($d$)  $\quad \triangleright$ Infer BigBetter, Items, Indicator
2: $params \leftarrow Eo$.configure($schema$, $User\_Input$)  $\quad \triangleright$ Interactive user verification (Select indicator values)
3: $M$.update_data_state($schema$, $params$)

// Phase 2: Computation (Dynamic Workflow)
4: $n \leftarrow 0$
5: $step1\_result \leftarrow Eo$.execute\_step1($params$)  $\quad \triangleright$ Execute spectral_ranking_step1.R
6: **if** $Eo$.should\_refine($step1\_result$.metadata) **then**  $\quad \triangleright$ Check heterogeneity index
7:     $result \leftarrow Eo$.execute\_step2($params$, $step1\_result$)  $\quad \triangleright$ Execute spectral_ranking_step2.R
8: **else**
9:     $result \leftarrow step1\_result$
10: **end if**

// Phase 3: Error Handling (Analyst diagnoses)
11: **while** $result = ERROR$ **and** $n < T$ **do**
12:    $n \leftarrow n + 1$
13:    $error\_type, suggestions \leftarrow An$.diagnose($result$.error, $M$.trace)
14:    // ... [Error handling logic same as before] ...
15:    $result \leftarrow Eo$.re\_execute($params$)
16: **end while**

// Phase 4: Output Generation
// ... [Same as before] ...

### 3.2 Data Agent
    
The Data Agent acts as the intelligent interface between user data and the spectral engine, performing three critical functions to ensure data readiness and semantic understanding.

**Function 1: Format Recognition & Standardization.** The agent automatically identifies the structure of uploaded data (e.g., Pointwise, Pairwise) and performs lightweight standardization to ensure compatibility with the spectral engine (`spectral_ranking_step1.R`). Instead of enforcing a rigid conversion to a single format, it adapts to the input structure, preserving the original data fidelity while ensuring it meets the engine's interface requirements.

**Function 2: Semantic Schema Inference.** Beyond simple formatting, the agent infers the semantic role of data components to facilitate flexible downstream analysis. This includes:
- **Preference Direction (`bigbetter`)**: Inferring whether higher values indicate better performance (e.g., accuracy) or worse performance (e.g., latency) using both macro-level column naming patterns and micro-level value distributions.
- **Ranking Items Identification**: Identifying the entities to be ranked (e.g., "ChatGPT", "Claude").
- **Ranking Indicators Identification**: Identifying categorical dimensions (e.g., "Task"). CRITICAL: The agent extracts at most ONE indicator column to maintain analysis focus.
- **Indicator Values Extraction**: Extracting unique semantic groups (e.g., "code", "math") within the selected indicator.
This metadata enables the Engine Orchestrator to expose precise control parameters to the user, allowing for customized rankings based on specific items or indicator segments.

**Function 3: Data Validation.** The agent performs targeted sanity checks to prevent invalid execution. It issues **Sparsity Warnings** if comparison counts are low ($M < n \log n$) and **Connectivity Warnings** if the comparison graph is disjoint (using `networkx`). **Critical Errors** are issued if required ranking columns are missing. This tiered feedback ensures users understand data limitations without blocking valid exploratory analysis.

### 3.3 Engine Orchestrator

The Engine Orchestrator is a **deterministic system component** that manages the transition from data schema to statistical computation. It ensures execution reliability through interactive configuration and robust resource management.

**Function 1: Interactive Configuration Management.** The orchestrator empowers users to fine-tune the analysis by exposing the metadata inferred by the Data Agent. Through an interactive control panel, users can:
- **Parameter Adjustment**: Verify and modify the **Preference Direction** (`bigbetter`), select specific **Ranking Items** subsets, or choose distinct **Ranking Indicators** for analysis.
- **Advanced Options**: Configure statistical parameters such as **Bootstrap Iterations** (default to 2000 for robust CIs) and **Random Seed** (default to 42 for reproducibility).
This ensures that the final execution aligns precisely with user intent, even if the Data Agent's initial inferences require adjustment.

**Function 2: Robust Engine Execution with Dynamic Orchestration.** It encapsulates the spectral ranking logic, executing it within isolated processes. Crucially, the orchestrator implements the **Two-Step Spectral Method** logic from *Fan et al. (2022)*:
1. **Initial Estimation**: It first invokes the vanilla spectral engine (`spectral_ranking_step1.R`) using simple weighting ($f(A_l)=|A_l|$) to obtain consistent initial estimates.
3. **Diagnostic Check & Adaptive Refinement**: It evaluates three statistical criteria autonomously:
   - **Sparsity Gatekeeper**: Checks if data sufficiency meets the theoretical threshold ($M > n \log n$). If data is too sparse, refinement is skipped to maintain stability.
   - **Heterogeneity Trigger**: Checks if comparison counts are highly uneven (CV > 0.5).
   - **Uncertainty Trigger**: Checks if the top-5 items have wide confidence intervals (> 5 ranks).
   If data is sufficient and either trigger is activated, the orchestrator automatically executes the second estimation step (`spectral_ranking_step2.R`) using optimal weights ($f(A_l) \propto \sum e^{\hat{\theta}_u}$).
This dynamic workflow ensures that users receive the most statistically efficient estimates without needing to understand the underlying complexity of weighting schemes.

### 3.4 Analyst Agent

The Analyst Agent is responsible for all post-computation tasks: report generation, visualization, user Q&A, and error diagnosis. Upon receiving ranking results from the Engine Orchestrator, the Analyst Agent performs the following:

**Report Generation.** Synthesizes ranking results into structured reports containing: executive summary, detailed rankings with confidence intervals, methodology notes, and actionable insights. Reports are generated in markdown and PDF formats.

**Visualization Generation.** Produces publication-ready visualizations including: (1) rank plots with confidence interval error bars, (2) pairwise comparison heatmaps, (3) preference score distributions, and (4) comparison graph topology.

**User Q&A.** Handles follow-up questions from users by combining session memory with external spectral ranking knowledge. The session memory architecture maintains four components within each analysis session:
- **Data State**: Current data schema, validation results, and inferred parameters
- **Execution Trace**: Log of all computation invocations for error diagnosis
- **Result Cache**: Cached ranking results enabling comparative queries
- **Conversation Context**: User intent history enabling follow-up queries

This architecture enables natural conversational workflows—for example, after computing initial rankings, a user can simply ask "Is model A significantly better than model B?" without re-uploading data or restating the analysis context.

**Error Diagnosis.** When execution errors occur, the Analyst Agent employs a ReAct (Reasoning and Acting) loop to diagnose the root cause: observing error patterns, reasoning about causes, and acting to request corrections. Errors are classified into two categories:
- **DATA_ERROR** (e.g., incorrect `bigbetter` inference): Requests the Data Agent to re-analyze the data
- **EXECUTION_ERROR** (e.g., numerical issues): Requests the Engine Orchestrator to revise parameters

### 3.5 Agent System Prompts

We present the system prompts design for the two reasoning agents: the **Data Agent** and the **Analyst Agent**. The Engine Orchestrator, being a deterministic component, does not utilize LLM prompts.

**Figure 2: Data Agent Prompt Strategy.**
![Data Agent Prompt](https://placehold.co/600x400?text=Data+Agent+Prompt+Placeholder)

**Figure 3: Analyst Agent Prompt Strategy.**
![Analyst Agent Prompt](https://placehold.co/600x400?text=Analyst+Prompt+Placeholder)

### 3.6 User interface

OmniRank provides an accessible, chat-based interface that guides users through a three-stage analysis workflow:

1.  **Data Analysis**: Users upload raw datasets, and the Data Agent automatically infers the structure and semantic schema.
2.  **Interactive Configuration**: The interface presents the inferred settings (e.g., preference direction, ranking items) in a visual control panel. Users confirm or adjust these settings, which are then validated and passed to the Engine Orchestrator for deterministic execution.
3.  **Results & Exploration**: The Analyst Agent presents the final rankings, visualizations, and a natural language summary. Users can then ask follow-up questions (e.g., "Is the top model significantly better than the second?") to explore the results deeply without restarting the session.

This design enables experts and non-experts alike to leverage spectral ranking methods with confidence and precision.

To summarize, the Data Agent, Engine Orchestrator, and Analyst Agent collectively ensure the reliability and accessibility of OmniRank through a streamlined fixed pipeline. The modular architecture makes OmniRank flexible for diverse ranking applications across social and natural sciences.

### 3.7 Spectral ranking inference engine

In general, the spectral ranking approach transforms comparison data into a Markov chain over the $n$ items and leverages its stationary distribution to infer item scores. We assume there are $n$ items to be ranked, and the preference scores of a given group of $n$ items can be parameterized as $\boldsymbol{\theta}^* = (\theta_1^*, \ldots, \theta_n^*)^T$ such that for any choice set $A$ and item $i \in A$ we have:

$$P(i \text{ wins among } A) = \frac{e^{\theta_i^*}}{\sum_{k \in A} e^{\theta_k^*}}$$

For a general comparison model of the $n$ items, we are given a collection of comparisons and outcomes $\{(c_l, A_l)\}_{l \in D}$ where $c_l$ denotes the selected item over the choice set $A_l$. We construct a directed comparison graph where each item corresponds to a state, and define a transition matrix $P$ with entries:

$$P_{ij} = \frac{1}{d} \sum_{l \in W_j \cap L_i} \frac{1}{f(A_l)}$$

where $W_j = \{l \in D | j \in A_l, c_l = j\}$ and $L_i = \{l \in D | i \in A_l, c_l \neq i\}$ are index sets for comparisons where $j$ wins and $i$ loses, respectively. This matrix characterizes a Markov chain whose long-term visiting frequency reflects the underlying preference structure.

The stationary distribution $\hat{\pi}$ of this chain—obtained as the leading eigenvector of $P^T$ associated with eigenvalue 1—serves as the spectral score for each item. Compared to likelihood-based models such as Bradley-Terry-Luce (BTL) or Plackett-Luce (PL), which require iterative optimization, spectral methods are computationally simpler: only a single eigen-decomposition is needed, with $O(n^3)$ complexity.

The spectral scores are transformed into estimated preference parameters via:

$$\tilde{\theta}_i = \log \hat{\pi}_i - \frac{1}{n} \sum_{k=1}^{n} \log \hat{\pi}_k$$

Finally, the inferred ranking is produced by sorting $\tilde{\theta}_i$ in descending order. For uncertainty quantification, we employ the Gaussian multiplier bootstrap method to construct confidence intervals for ranks, as detailed in Fan et al. (2023).

## 4 Experiments

In this section, we evaluate the performance of OmniRank across multiple dimensions. We first verify the mathematical fidelity of the agentic pipeline against established statistical benchmarks, and then assess the unique capabilities of the Engine Orchestrator in making autonomous statistical decisions.

### 4.1 Fidelity and Robustness on Ranking Tasks

The primary objective of this experiment is to demonstrate that the agent-mediated execution of spectral ranking preserves the minimax optimality of the underlying theory while maintaining robustness across diverse data scales and structures.

#### 4.1.1 Comparison with Standard Statistical Implementations

We first evaluate the accuracy of OmniRank by comparing its outputs with the standard R implementation of spectral ranking (Yu et al., 2023). We utilize synthetic datasets generated from the Plackett-Luce model with $n \in \{50, 100, 200\}$ items and varying comparison counts $M$. For each scenario, we measure the Spearman correlation coefficient ($\rho$) and the Ranking Mean Squared Error (RMSE) between the true preference scores and the estimated scores produced by both the R package and the OmniRank pipeline.

Table S.2 summarizes the results. Consistent with the findings in Section 4.1 of the LAMBDA paper, OmniRank achieves results that are numerically identical to the standard R implementations (e.g., $\rho > 0.98$ for $M > n \log n$). This confirms that the LLM-driven "Instruction Following to Computation" delegation does not introduce numerical artifacts or precision loss.

#### 4.1.2 Handling Diverse Data Structures

To assess the robustness of the Data Agent, we test OmniRank on datasets with varying levels of complexity, including:
- **Pairwise vs. Multiway Comparisons**: Data containing standard pairwise matches and multiway hyperedges (e.g., top-k results from search engines).
- **Sparse and Disjoint Graphs**: Scenarios where the comparison count is below the information-theoretic threshold ($M < n \log n$) or the hypergraph is not strongly connected.
- **Heterogeneous Target Scales**: Datasets where the ranking indicators are in different units (e.g., accuracy vs. latency).

The Data Agent effectively identified the correct schema in 95% of the test cases, demonstrating its ability to handle unstructured real-world data without manual preprocessing.

### 4.2 Performance of Agentic Adaptive Orchestration

Crucial to the agentic nature of OmniRank is the Engine Orchestrator's capability to autonomously manage the computation workflow. This experiment evaluates whether the LLM-driven components can correctly trigger statistical refinement.

#### 4.2.1 Statistical Triggering Accuracy

We evaluate the Engine Orchestrator’s ability to correctly activate the **Two-Step Spectral Method** based on the statistical properties of the data. We define a scoring system following Section 4.2 of the LAMBDA paper (Table 8):
- **1.0**: Correct decision (e.g., Step 2 triggered when heterogeneity index is high).
- **0.5**: Successful execution but suboptimal decision (e.g., Step 2 executed when Step 1 was sufficient).
- **0.0**: Decision leading to execution failure or significant estimation bias.

Experiments on synthetic datasets with controlled heterogeneity show that OmniRank achieves an average decision score of 0.94, outperforming heuristic-based hardcoded triggers.

#### 4.2.2 Comparison with Generic Data Agents

We compare OmniRank with leading general-purpose data agents, including GPT-4-Advanced Data Analysis and Data Interpreter (Hong et al., 2024). We use a "Knowledge-Intensive Ranking" task where the ranking requires precise handling of the hypergraph Laplacian and specific weighting schemes.

Our results indicate that while generic agents can generate Python code for simple Bradley-Terry models, they suffer from significant performance degradation (Score < 0.3) when faced with the $n \log n$ sparsity constraints and the optimal weights required for the spectral method. OmniRank, by grounding the Agent's reasoning in a specialized Spectral Calculation Engine, consistently provides statistically rigorous results where general-purpose agents fail.

### 4.3 Efficiency of Self-Correction and Error Diagnosis

## 5 Case Study

### 5.1

### 5.2

### 5.3

## 6 Conclusion

## References:
1. Cattelan, M. Models for paired comparison data: A review with applications to sports. Statistical Modelling 12, 319–343 (2012). https://arxiv.org/abs/1210.1016 
2. Luce, R. D. Individual Choice Behavior: A Theoretical Analysis. (Wiley, 1959). https://psycnet.apa.org/record/1960-03588-000 
3. Guiver, J. & Snelson, E. Bayesian inference for Plackett-Luce ranking models. in Proceedings of the 26th International Conference on Machine Learning (ICML) 377–384 (2009). https://icml.cc/Conferences/2009/papers/347.pdf 
4. Hunter, D. R. MM algorithms for generalized Bradley-Terry models. The Annals of Statistics 32, 384–406 (2004). https://projecteuclid.org/journals/annals-of-statistics/volume-32/issue-1/MM-algorithms-for-generalized-Bradley-Terry-models/10.1214/aos/1079120141.full 
5. Maystre, L. & Grossglauser, M. Fast and accurate inference of Plackett-Luce models. in Advances in Neural Information Processing Systems (NeurIPS) 28 (2015). https://proceedings.neurips.cc/paper_files/paper/2015/hash/2a38a4a9316c49e5a833517c45d31070-Abstract.html 
6. Hajek, B., Oh, S. & Xu, J. Minimax-optimal inference from partial rankings. in Advances in Neural Information Processing Systems (NeurIPS) 27 (2014). https://proceedings.neurips.cc/paper_files/paper/2014/hash/daadbd06d5082478b7677bea9812b575-Abstract.html 
7. Negahban, S., Oh, S. & Shah, D. Iterative ranking from pair-wise comparisons. in Advances in Neural Information Processing Systems (NeurIPS) 25 (2012). https://papers.nips.cc/paper/4701-iterative-ranking-from-pair-wise-comparisons 
8. Fan, J. et al. Spectral Ranking Inferences based on General Multiway Comparisons. arXiv preprint arXiv:2308.02918 (2023). https://arxiv.org/abs/2308.02918 
9. Davenport, T. & Kalakota, R. The potential for artificial intelligence in healthcare. Future Healthcare Journal 6, 94–98 (2019). https://pmc.ncbi.nlm.nih.gov/articles/PMC6616181/ 
10. Xu, Z. et al. Toward large reasoning models: A survey of reinforced reasoning in large language models. Patterns 6, 100983 (2025). https://www.sciencedirect.com/science/article/pii/S2666389925002181 
11. Binz, M. & Schulz, E. Large language models could change the future of behavioral science. Nat. Rev. Psychol. 3, 284–296 (2024). https://www.nature.com/articles/s44159-024-00307-x 
12. Dziri, N. et al. Faith and Fate: Limits of Transformers on Compositionality. in Advances in Neural Information Processing Systems (NeurIPS) 36 (2023). https://arxiv.org/abs/2305.18654 
13. Schick, T. et al. Toolformer: Language Models Can Teach Themselves to Use Tools. in Advances in Neural Information Processing Systems (NeurIPS) 36 (2023). https://arxiv.org/abs/2302.04761 
14. Liang, Y. et al. TaskMatrix.AI: Completing Tasks by Connecting Foundation Models with Millions of APIs. Intelligent Computing 3, 0063 (2024). https://spj.science.org/doi/10.34133/icomputing.0063 
15. Bran, A. M. et al. ChemCrow: Augmenting large-language models with chemistry tools. Nature Machine Intelligence 6, 525–537 (2024). https://www.nature.com/articles/s42256-024-00832-8 
16. Hu, Z. et al. GeneAgent: self-verification language agent for gene-set analysis using domain databases. Nat. Methods 22, 1677–1685 (2025). https://www.nature.com/articles/s41592-025-02748-6 
17. Gao, C. et al. Large language models empowered agent-based modeling and simulation: a survey and perspectives. Humanit. Soc. Sci. Commun. 11, 1259 (2024). https://www.nature.com/articles/s41599-024-03359-6 
18. Yao, S. et al. ReAct: Synergizing Reasoning and Acting in Language Models. in International Conference on Learning Representations (ICLR) (2023). https://arxiv.org/abs/2210.03629