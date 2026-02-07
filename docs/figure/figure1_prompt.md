# Figure 1 Prompt: OmniRank System Architecture

Create a detailed scientific architecture diagram illustrating the OmniRank agentic AI system for statistically rigorous ranking inference from arbitrary multiway comparisons. The diagram should be in a clean, professional scientific illustration style similar to figures in Nature Methods papers, using simple geometric shapes, clean lines, and minimal icons.

**CRITICAL STYLE REQUIREMENTS:**
- Use only simple geometric shapes (rectangles, circles, arrows, lines)
- All icons should be abstract/schematic (like scientific flowchart symbols), NOT realistic illustrations
- Use flat design with no gradients, shadows, or 3D effects
- Text labels should be clean sans-serif font
- Overall style: hand-drawn scientific diagram aesthetic, similar to LaTeX/TikZ generated figures
- DO NOT include any figure title or caption in the diagram itself

**Aspect Ratio:** 3:4 (portrait orientation, suitable for journal figure)

**Color Scheme:**
- Deep blue (#1E40AF) for LLM-powered agents (Data Agent, Analyst Agent)
- Emerald green (#059669) for deterministic computation (Engine Orchestrator, R Scripts)
- Amber/Gold (#D97706) for user interaction elements
- Light purple (#7C3AED) for data flow arrows
- Coral red (#DC2626) for feedback/error loops (dashed arrows)
- White background, no gradients

---

**LAYOUT STRUCTURE (Top to Bottom):**

The diagram is organized vertically in 5 horizontal bands:

**BAND 1 (Top): User Input & Data Upload**
**BAND 2: Data Agent - LLM-Powered Data Understanding**
**BAND 3: Engine Orchestrator - Spectral Ranking Execution**
**BAND 4: Analyst Agent - Result Interpretation & Interaction**
**BAND 5 (Bottom): User Output & Interactive Chat**

---

**DETAILED PANEL DESCRIPTIONS:**

**Panel a - User Input Interface (top-center):**
- Draw a simple user icon (circle with person silhouette)
- File upload symbol below (document with upward arrow)
- Small labels showing supported formats: "CSV", "Excel"
- Arrow labeled "Raw Comparison Data" pointing downward to Data Agent
- Text annotation showing data types:
  - "Pointwise: Model × Task scores"
  - "Pairwise: Head-to-head results"
  - "Multiway: Top-k rankings"

**Panel b - Data Agent (second band, full width, BLUE accent):**
- Large rounded rectangle with blue border, labeled "Data Agent" at top
- Inside the rectangle, show three connected sub-modules:

  LEFT sub-module: Format Recognition
  - Magnifying glass icon over table
  - Label: "Format Recognition"
  - Output classification: "pointwise / pairwise / multiway / invalid"
  
  CENTER sub-module: Validation Engine
  - Checkmark in shield icon
  - Label: "Validation"
  - Three validation checks displayed:
    - "Sparsity: M ≥ n·log(n)"
    - "Connectivity: Graph connected?"
    - "Data Integrity"
  
  RIGHT sub-module: Schema Inference
  - Lightbulb icon
  - Label: "Schema Inference"
  - Extracted fields: "BigBetter", "Ranking Items", "Indicator Column"

- Output arrow pointing down: "Validated Schema"
- Dashed red feedback arrow looping back labeled "Validation Feedback"

**Panel c - Engine Orchestrator (third band, full width, GREEN accent):**
- Large rectangular container with green border, labeled "Engine Orchestrator" at top
- CRITICAL: Show a simple LEFT-TO-RIGHT flowchart with the following layout:

  ```
  [Input] → [Spectral Ranking Engine] → [Output]
  ```

  BLOCK 1 (left): Input Processing
  - Rectangle showing data preprocessing
  - Label above: "Data Preprocessing"
  - Features: "Item Selection", "Indicator Filtering"
  
  BLOCK 2 (center): Spectral Ranking Engine
  - Rectangle with R logo icon inside
  - Label above: "Spectral Ranking"
  - Formula below: "f(Aₗ) = |Aₗ|"
  - Small text: "Bootstrap: 2000 iter"
  
  BLOCK 3 (right): Output
  - Final output arrow points DOWN to Analyst Agent, labeled "Ranking Results + CI"

- Input arrow from Data Agent enters from TOP-LEFT: "Schema + Data Path"

**Panel d - Analyst Agent (fourth band, full width, BLUE accent):**
- Large rounded rectangle with blue border, labeled "Analyst Agent"
- Inside, show three functional modules side by side:

  LEFT module: Report Generation
  - Document icon with text lines
  - Label: "Report Synthesis"
  - Report sections listed:
    - "Executive Summary"
    - "Statistical Significance"
    - "Performance Gaps"
    - "Methodology Notes"
    - "Recommendations"
  
  CENTER module: Visualization Data
  - Chart icon
  - Label: "Visualization Suite"
  - Three visualization types: "Forest Plot", "Heatmap", "Distribution"
  - Small note: "(rendered by frontend)"
  
  RIGHT module: Interactive Q&A
  - Chat bubble icon
  - Label: "Q&A Interface"
  - Features: "CI interpretation", "Statistical ties detection", "Domain insights"

- Input arrow from Orchestrator: "Ranking Results"
- Dashed red feedback arrow to Orchestrator labeled "Error Diagnosis"
- Output arrow pointing down: "Report + Visualizations + Answers"

**Panel e - User Output (bottom band):**
- User icon receiving three types of output:
  - Report document icon: "Analysis Report"
  - Chart icon: "Interactive Visualizations"
  - Chat bubble icon: "Follow-up Q&A"
- Bidirectional arrows showing conversation flow
- Small annotation: "Session Memory: Data State, Execution Trace, Conversation History"

---

**VISUAL FLOW ANNOTATIONS:**
- Use solid arrows (→) with light purple color for main data/workflow flow
- Use dashed arrows (-->) with coral red color for feedback/error loops
- Number each panel clearly with letters (a, b, c, d, e) in blue circles
- Keep arrow styles consistent throughout
- Show the conditional branching in Panel c clearly with Yes/No labels

**KEY VISUAL ELEMENTS TO EMPHASIZE:**
1. The distinction between LLM-powered components (blue, rounded) and deterministic components (green, angular)
2. The streamlined spectral ranking execution flow
3. Feedback loops for validation and error handling
4. The three-function structure of each agent (Data Agent: 3 sub-modules, Analyst Agent: 3 modules)
5. Session Memory as a shared resource (shown at bottom)

**TYPOGRAPHY:**
- Panel labels (a, b, c, d, e): Bold, positioned in blue circles
- Component names: Bold, 12pt
- Sub-module labels: Regular, 10pt
- Annotations and formulas: Italic, 9pt
- Mathematical notation: Use standard symbols (θ̂, Σ, ∝, ≥)

**OVERALL AESTHETIC:**
- Clean, minimal, professional scientific figure style
- Similar to TikZ/LaTeX-generated diagrams in Nature Methods
- No photorealistic elements or AI-generated artifacts
- Portrait orientation (3:4 ratio) optimized for single-column print figure
- Clear visual hierarchy: User Input → Data Agent → Engine Orchestrator → Analyst Agent → User Output
- Emphasis on the "decoupled reasoning" paradigm: LLM for semantic understanding, R scripts for computation
