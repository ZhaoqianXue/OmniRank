# Figure 5 Prompt: Analyst Agent System Prompt Structure

Create a detailed scientific diagram illustrating the three-layer structure of the Analyst Agent's system prompt in OmniRank. The diagram should be in a clean, professional scientific illustration style similar to figures in Nature Methods papers, showing the hierarchical organization of prompt components with emphasis on spectral ranking domain knowledge.

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
- Violet purple (#8B5CF6) for Layer 3 (Knowledge) - different from Data Agent to distinguish
- Light gray (#F3F4F6) for content boxes
- White background, no gradients

---

**LAYOUT STRUCTURE (Top to Bottom):**

The diagram shows a stacked layer architecture:

**HEADER: System Prompt Container**
**LAYER 1: Role Specification**
**LAYER 2: Operational Constraints**
**LAYER 3: Spectral Ranking Knowledge**

---

**DETAILED PANEL DESCRIPTIONS:**

**HEADER - System Prompt Container:**
- Large rounded rectangle with blue border encompassing all layers
- Header bar at top: "Analyst Agent System Prompt"
- Small LLM icon (brain/chip symbol) in corner
- Annotation: "RAG-Enhanced Interpretation"

**LAYER 1 - Role Specification (top layer, LIGHT BLUE):**
- Rounded rectangle with light blue background
- Header: "Layer 1: Role Specification"
- Icon: Analyst/consultant symbol (person with chart)
- Content box showing:
  ```
  "You are an expert statistical analyst
   specializing in ranking inference.
   Your task is to generate insightful,
   professional analysis reports."
  ```
- Key responsibilities listed:
  - "Report Synthesis"
  - "Interactive Q&A"
  - "Error Diagnosis"
- Annotation arrow: "Defines consultant persona"

**LAYER 2 - Operational Constraints (middle layer, GREEN):**
- Rounded rectangle with green background
- Header: "Layer 2: Operational Constraints"
- Icon: Document template symbol
- Content organized in two columns:

  LEFT column - Report Structure:
  - "Required Sections:"
    1. "Executive Summary"
    2. "Statistical Significance"
    3. "Performance Gaps"
    4. "Methodology Notes"
    5. "Recommendations"
  - "Length: 500-800 words"
  
  RIGHT column - Response Rules:
  - "Q&A Format: BLUF method"
    - "Direct answer first"
    - "Evidence with data"
    - "Plain language explanation"
  - "Max 3 paragraphs per answer"
  - "Reference actual item names"

- Annotation arrow: "Ensures structured, actionable output"

**LAYER 3 - Spectral Ranking Knowledge (bottom layer, largest, VIOLET):**
- Rounded rectangle with violet background (largest layer, visually prominent)
- Header: "Layer 3: Spectral Ranking Domain Knowledge"
- Icon: Graph/network symbol
- Content organized in FOUR knowledge blocks:

  BLOCK A - Core Logic:
  - Box titled "Ranking Intuition"
  - Content:
    - "Think of spectral ranking as 'voting'"
    - "Beating strong opponents worth more"
    - "θ reflects global consensus power"
  - Simple diagram: nodes with arrows showing vote flow

  BLOCK B - Confidence Intervals:
  - Box titled "CI Interpretation"
  - Content:
    - "95% CI: range of plausible true ranks"
    - "Overlap = Statistical tie"
    - "No overlap = Significant difference"
  - Small visual: two error bars, one overlapping, one not

  BLOCK C - Data Quality:
  - Box titled "Quality Metrics"
  - Two metrics explained:
    - "Sparsity Ratio ≥ 1.0: Clear picture"
    - "Sparsity Ratio < 1.0: Grainy image"
    - "High Heterogeneity: Uneven match counts"
  - Analogy: "Like pixel density in an image"

  BLOCK D - Bootstrap Methods:
  - Box titled "Uncertainty Quantification"
  - Content:
    - "Bootstrap: 2000 iterations default"
    - "95% CI: confidence interval for ranks"
    - "Gaussian multiplier bootstrap"
  - Small visual: confidence bar with whiskers

- Annotation arrow: "Enables expert reasoning without fine-tuning"

---

**VISUAL ANNOTATIONS:**

**Layer Hierarchy Indicator (left side):**
- Vertical arrow pointing downward
- Labels at each layer level:
  - "Persona (WHO)"
  - "Format (HOW)"
  - "Expertise (WHAT)"

**Communication Guidelines Callout (right side):**
- Small annotation box titled "Plain Language Rules":
  - "Avoid: 'stationary distribution', 'Markov chain'"
  - "Use: 'statistical tie', 'clear winner'"
  - "Focus: 'Is A better than B?'"

**Comparison with Data Agent (bottom annotation):**
- Small comparison table:
  ```
  | Aspect       | Data Agent      | Analyst Agent      |
  |--------------|-----------------|-------------------|
  | Focus        | Data structure  | Result meaning    |
  | Knowledge    | Format rules    | CI interpretation |
  | Output       | JSON schema     | Natural language  |
  ```

**KEY VISUAL ELEMENTS TO EMPHASIZE:**
1. The three-layer stack structure (Role → Constraints → Knowledge)
2. Layer 3 (Knowledge) is visually largest and uses different color (violet vs amber)
3. Knowledge blocks include visual aids (error bars, flow arrows)
4. Plain language translation examples prominent
5. Clear distinction from Data Agent's knowledge focus

**TYPOGRAPHY:**
- Layer headers: Bold, 12pt
- Content text: Regular, 10pt
- Analogies: Italic, emphasized
- Technical terms in quotes when showing what to avoid
- Greek symbols (θ) in math font

**OVERALL AESTHETIC:**
- Clean, minimal, professional scientific diagram
- Similar to Figure 4 but with distinct color (violet for Analyst)
- Portrait orientation (3:4 ratio)
- Clear visual hierarchy showing prompt composition
- Emphasis on translating statistical concepts to plain language
