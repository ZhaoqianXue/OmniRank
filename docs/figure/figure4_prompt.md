# Figure 4 Prompt: Data Agent System Prompt Structure

Create a detailed scientific diagram illustrating the three-layer structure of the Data Agent's system prompt in OmniRank. The diagram should be in a clean, professional scientific illustration style similar to figures in Nature Methods papers, showing the hierarchical organization of prompt components.

**CRITICAL STYLE REQUIREMENTS:**
- Use only simple geometric shapes (rectangles, rounded rectangles, arrows)
- All icons should be abstract/schematic, NOT realistic illustrations
- Use flat design with no gradients, shadows, or 3D effects
- Text labels should be clean sans-serif font
- Overall style: hand-drawn scientific diagram aesthetic, similar to LaTeX/TikZ generated figures
- DO NOT include any figure title or caption in the diagram itself

**Aspect Ratio:** 3:4 (portrait orientation, suitable for journal figure)

**Color Scheme:**
- Deep blue (#1E40AF) for the main prompt container
- Light blue (#93C5FD) for Layer 1 (Role)
- Emerald green (#059669) for Layer 2 (Constraints)
- Amber/Gold (#D97706) for Layer 3 (Knowledge)
- Light gray (#F3F4F6) for content boxes
- White background, no gradients

---

**LAYOUT STRUCTURE (Top to Bottom):**

The diagram shows a stacked layer architecture:

**HEADER: System Prompt Container**
**LAYER 1: Role Specification**
**LAYER 2: Operational Constraints**
**LAYER 3: Domain Knowledge**

---

**DETAILED PANEL DESCRIPTIONS:**

**HEADER - System Prompt Container:**
- Large rounded rectangle with blue border encompassing all layers
- Header bar at top: "Data Agent System Prompt"
- Small LLM icon (brain/chip symbol) in corner
- Annotation: "Structured System Instructions Pattern"

**LAYER 1 - Role Specification (top layer, LIGHT BLUE):**
- Rounded rectangle with light blue background
- Header: "Layer 1: Role Specification"
- Icon: Person with badge/ID symbol
- Content box showing:
  ```
  "You are a Data Schema Analyst for OmniRank,
   an expert in understanding structured data
   for spectral ranking analysis."
  ```
- Key responsibilities listed:
  - "Format Recognition"
  - "Standardization Assessment"
- Annotation arrow pointing to content: "Defines agent identity"

**LAYER 2 - Operational Constraints (middle layer, GREEN):**
- Rounded rectangle with green background
- Header: "Layer 2: Operational Constraints"
- Icon: Checklist/clipboard symbol
- Content organized in two columns:

  LEFT column - Output Format:
  - "JSON response format"
  - "Required fields:"
    - "format"
    - "engine_compatible"
    - "bigbetter"
    - "ranking_items"
    - "indicator_col"
  
  RIGHT column - Processing Rules:
  - "No markdown in output"
  - "Single indicator column max"
  - "Cardinality constraints: 2-20 values"

- Annotation arrow: "Ensures consistent, parseable output"

**LAYER 3 - Domain Knowledge (bottom layer, largest, AMBER):**
- Rounded rectangle with amber background (largest layer)
- Header: "Layer 3: Domain Knowledge"
- Icon: Book/knowledge base symbol
- Content organized in THREE sections:

  SECTION A - Format Definitions:
  - Box titled "Data Formats"
  - Three format cards:
    - "Pointwise: Dense numeric matrix, items as columns"
    - "Pairwise: Sparse 0/1 matrix, head-to-head comparisons"
    - "Multiway: Rank positions (1st, 2nd, 3rd...)"
  - "Invalid: <2 items, non-numeric, empty"

  SECTION B - Validation Thresholds:
  - Box titled "Spectral Theory Rules"
  - Key thresholds:
    - "Sparsity: M ≥ n·log(n)"
    - "Connectivity: Single connected component"
    - "Minimum items: n ≥ 2"
  - Source citation: "[Fan et al., 2026]"

  SECTION C - Inference Heuristics:
  - Box titled "BigBetter Inference"
  - Two columns:
    - "Higher is better: score, accuracy, f1, auc, win"
    - "Lower is better: error, loss, time, latency, cost"
  - "Default to engine_compatible=true"

- Annotation arrow: "Embeds expert knowledge without fine-tuning"

---

**VISUAL ANNOTATIONS:**

**Layer Hierarchy Indicator (left side):**
- Vertical arrow pointing downward
- Labels at each layer level:
  - "Identity (WHO)"
  - "Rules (HOW)"
  - "Expertise (WHAT)"

**Design Principle Callouts (right side):**
- Three small annotation boxes:
  - "Separation of concerns"
  - "In-context learning"
  - "No model modification required"

**KEY VISUAL ELEMENTS TO EMPHASIZE:**
1. The three-layer stack structure (Role → Constraints → Knowledge)
2. Layer 3 (Knowledge) is visually largest, emphasizing its importance
3. Each layer has distinct color coding
4. Content boxes show actual prompt excerpts (not paraphrased)
5. Annotations explain the purpose of each layer

**TYPOGRAPHY:**
- Layer headers: Bold, 12pt
- Content text: Regular, 10pt (monospace for code/JSON)
- Annotations: Italic, 9pt
- Prompt excerpts: Quoted text style

**OVERALL AESTHETIC:**
- Clean, minimal, professional scientific diagram
- Similar to software architecture diagrams in NeurIPS papers
- Portrait orientation (3:4 ratio)
- Clear visual hierarchy showing prompt composition
- Emphasis on the "Structured System Instructions" pattern from OpenAI best practices
