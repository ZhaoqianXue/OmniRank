# Figure Prompt: OmniRank System Architecture

Use this prompt to generate a detailed scientific system architecture diagram for OmniRank. The diagram should meet Nature/JASA publication aesthetic standards; show exactly ONE LLM agent (gpt-5.4-nano) and all 10 tools; use strictly 2D flat style (no 3D effects, no shadows); and use a polished, professional style with labeled arrows and refined color palette.

---

## Prompt for Image Generation

Create a detailed scientific system architecture diagram illustrating the OmniRank agentic framework for spectral ranking inference. The diagram must meet the high aesthetic standards of top-tier journals (Nature, JASA, PNAS): polished, publication-ready, with refined visual hierarchy and professional polish. **Style requirements**: (1) Strictly 2D flat graphics—no 3D effects, no shadows, no gradients that create depth; use flat colors and simple borders. (2) Exactly ONE LLM agent icon in the entire diagram—do not duplicate the agent across phases. (3) Software/ML system architecture diagram (not medical illustration), like a flowchart from a CS/ML research paper, with labeled arrows showing data flow and cause-and-effect relationships.

### Key Elements to Include

**Central focus: Three-phase pipeline layout**
- A horizontal or vertical flow showing the three phases: (1) Data Processing and Schema Inference, (2) Interactive Configuration and Computation, (3) Result Synthesis and User Interaction.
- Each phase should be visually distinct (e.g., rounded rectangles or bordered regions) with clear phase labels.

**LLM Agent role**
- **Critical: Only ONE agent instance**—OmniRank uses a single LLM agent operating within one context window. Depict the agent exactly once (e.g., as a central spine, or one node that receives input and orchestrates all phases). Do NOT duplicate the agent icon across phases; do NOT show multiple "gpt-5.4-nano" or brain icons. The same agent orchestrates Phase 1, Phase 2, and Phase 3 sequentially.
- Depict this single agent as a central orchestrator that receives user requests (natural language + data upload), interprets intent, and selects tools based on the current pipeline phase.
- **Mandatory: Display the model name prominently**—label the agent as "LLM Agent (gpt-5.4-nano)" or place a visible badge/tag (e.g., "gpt-5.4-nano") on or beside the agent node so the backbone model is immediately identifiable.
- Show the agent as a "cognitive controller" that handles semantic understanding (data interpretation, schema inference, natural-language synthesis) but delegates all mathematical computation to external tools.
- Use a distinct icon or shape (e.g., brain/controller icon) to distinguish the agent from deterministic tools—but only one such icon in the entire diagram.

**Tool registry (10 tools in 4 categories)**
- **Mandatory: Visually distinguish all tools**—each tool must be explicitly labeled with a "Tool" badge, wrench icon, or consistent "Tool N" notation (e.g., Tool 1–10) so viewers can immediately identify which components are tools vs. the agent or engine.
- **Data Tools (5)** — label as "Data Tools" with a category header; list each with its tool name: `read_data_file`, `infer_semantic_schema`, `validate_data_format`, `validate_data_quality`, `preprocess_data`—arranged as a cluster or row, with arrows from agent to each tool and back (tool outputs feed agent context).
- **User Interaction Tool (1)** — label as "Tool: request_user_confirmation"; shown as a human-in-the-loop checkpoint (e.g., user icon or handshake) between Phase 1 and Phase 2, with a dashed boundary indicating the explicit confirmation step before computation proceeds.
- **Engine Tool (1)** — label as "Tool: execute_spectral_ranking"; depict as a distinct "Spectral Calculation Engine" block (e.g., R script icon or matrix/eigenvalue symbol) that runs in an isolated subprocess; show inputs (validated data, confirmed schema, bootstrap params) and outputs (preference scores, ranks, confidence intervals). **Critical**: Only this tool invokes the Spectral Engine; the five Data Tools (including `preprocess_data`) do NOT connect directly to the engine—they feed the agent, which then calls the engine via `execute_spectral_ranking` after user confirmation.
- **Analysis Tools (3)** — label as "Analysis Tools" with a category header; list each: `generate_visualizations`, `generate_report`, `answer_question`. Show the output flow: R → generate_visualizations → Plots; R + Plots → generate_report → Report (one Report only); R + Report → answer_question → Q&A. Do NOT show Report twice.

**Decoupling principle**
- Illustrate the separation between "LLM reasoning" (semantic, stochastic) and "deterministic computation" (spectral engine, validation, visualization)—e.g., use different background tints (e.g., light blue for LLM domain, light gray for deterministic domain) or a dashed boundary.
- Show that numerical operations (eigenvector computation, bootstrap, graph connectivity checks) are executed by verified tools, not by the LLM internally—use a "no computation" or "delegation" symbol (e.g., arrow with "delegates to" label) from agent to engine.

**Data flow**
- **Input**: User uploads dataset D (CSV/Excel) and natural language request; arrow from "User" to agent.
- **Phase 1 flow**: `read_data_file` → summary → `infer_semantic_schema` → schema S; then validation loop: `validate_data_format` → (if fixable) `preprocess_data` → re-validate until ready; then `validate_data_quality` → Vq.
- **Phase 2 flow**: Phase 2 inputs come from Phase 1's complete validation output, not from a single tool. `request_user_confirmation` receives (S, Vf, Vq) from Phase 1 and outputs (Sc, B, seed); then `execute_spectral_ranking` receives validated data D (aggregated from Phase 1), confirmed schema Sc, bootstrap B, and seed → ranking results R.
- **Phase 3 flow**: Ranking results R flow to all three Analysis Tools. **Exact sequence**: (1) `generate_visualizations` (R) → plots; (2) `generate_report` (R, plots) → **one report only** (report incorporates both R and plots); (3) `answer_question` (query, R, report) → interactive Q&A loop (circular arrow back to user). **Critical**: Show Report exactly once—as the sole output of `generate_report`. Do NOT duplicate the Report icon or show Report before and after Charts/Plots; the correct order is Plots first, then the single Report, then Q&A.

**Spectral engine internals (optional detail)**
- If space permits, show a simplified view inside the engine: "Hypergraph construction" → "Transition matrix P" → "Eigenvector / stationary distribution" → "Preference scores θ̂" → "Gaussian multiplier bootstrap" → "Rank confidence intervals [R̂^L, R̂^U]".

**Before-and-after comparison (optional)**
- Split the diagram into "Without OmniRank" (generic LLM attempting ranking directly—red tones, "hallucination" or "ad hoc heuristics" labels) and "With OmniRank" (tool-calling pipeline—blue/green tones, "deterministic spectral inference" label) to highlight the architectural benefit.

**Labels and annotations**
- Use clear text labels for key components (e.g., "LLM Agent (gpt-5.4-nano)", "Spectral Engine", "Human-in-the-Loop", "Data Tools", "Analysis Tools").
- Arrows indicating data flow (solid) vs. control flow (dashed); label critical arrows (e.g., "delegates computation", "validated data", "ranking results").
- Include a legend explaining symbols (e.g., solid arrow = data flow, dashed = control, tool badge = tool component).

**Overall aesthetic — Nature/JASA publication quality**
- **Strictly 2D, flat style**: No 3D effects. No shadows, no drop shadows, no gradients that create depth or embossed/raised appearance. Use flat colors, flat fills, and simple borders. Create hierarchy through flat color blocks, line weight, and layout—not through shadows or 3D effects.
- **Refined color palette**: Sophisticated blues and grays with accent tones (e.g., slate, navy, soft teal); avoid harsh primaries. Use flat, solid fills for regions; avoid gradient backgrounds that suggest depth.
- **Typography**: Clean, professional sans-serif for labels; ensure model name "gpt-5.4-nano" and tool names are legible and prominent.
- **Phase separation**: Give each phase a distinct visual treatment using flat borders, flat background tints, or flat dividers—no shadows or gradients.
- High-resolution, vector-style graphics suitable for print; no text-heavy clutter; keep it visually intuitive while meeting top-journal aesthetic standards.

---

## Reference: Caption for Writing

**[Figure: OmniRank system architecture]** OmniRank comprises a single LLM agent and a fixed registry of ten tools. The agent orchestrates a three-phase pipeline: data processing (five data tools), computation (one engine tool with an explicit user confirmation step), and output generation (three analysis tools). All orchestration occurs within a single context window; numerical computation is delegated to deterministic tools.
