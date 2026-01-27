# Figure 2 Prompt: Data Agent Validation Workflow

Create a detailed scientific flowchart illustrating the Data Agent's hierarchical validation process in OmniRank. The diagram should be in a clean, professional scientific illustration style similar to figures in Nature Methods papers, using simple geometric shapes, clean lines, and standard flowchart symbols.

**CRITICAL STYLE REQUIREMENTS:**
- Use only standard flowchart symbols (rectangles for processes, diamonds for decisions, rounded rectangles for start/end)
- All icons should be abstract/schematic, NOT realistic illustrations
- Use flat design with no gradients, shadows, or 3D effects
- Text labels should be clean sans-serif font
- Overall style: hand-drawn scientific diagram aesthetic, similar to LaTeX/TikZ generated figures
- DO NOT include any figure title or caption in the diagram itself

**Aspect Ratio:** 16:9 (portrait orientation, suitable for journal figure)

**Color Scheme:**
- Deep blue (#1E40AF) for LLM-powered processes
- Coral red (#DC2626) for error/rejection paths
- Amber/Gold (#D97706) for warning paths
- Emerald green (#059669) for success/valid paths
- Light gray (#E5E7EB) for process boxes
- White background, no gradients

---

**LAYOUT STRUCTURE (Top to Bottom):**

The flowchart is organized vertically showing the hierarchical validation process with three severity levels:

**LEVEL 1: Critical Errors (blocking)**
**LEVEL 2: Warnings (non-blocking)**
**LEVEL 3: Valid Data (proceed to schema inference)**

---

**DETAILED FLOWCHART ELEMENTS:**

**START (top-center):**
- Rounded rectangle
- Label: "Raw Data Input"
- Arrow pointing down to first decision

**DECISION 1 - Data Integrity Check (CRITICAL):**
- Diamond shape with red border
- Label inside: "Data Readable?"
- Checks: "Valid CSV/Excel?", "Has rows?", "Has columns?"
- TWO outputs:
  - NO (red arrow pointing RIGHT): → "REJECT: Format Error" (red rounded rectangle, terminal)
  - YES (green arrow pointing DOWN): → Next decision

**DECISION 2 - Minimum Items Check (CRITICAL):**
- Diamond shape with red border
- Label inside: "Items ≥ 2?"
- Check: "At least 2 rankable items exist?"
- TWO outputs:
  - NO (red arrow pointing RIGHT): → "REJECT: Insufficient Items" (red rounded rectangle, terminal)
  - YES (green arrow pointing DOWN): → Next decision

**DECISION 3 - Connectivity Check (WARNING):**
- Diamond shape with amber border
- Label inside: "Graph Connected?"
- Check: "Comparison graph forms single connected component?"
- TWO outputs:
  - NO (amber arrow pointing RIGHT): → "WARNING: Disconnected Subgraphs" (amber rectangle)
    - Small annotation: "Rankings computed within largest component only"
    - Arrow continues DOWN (non-blocking)
  - YES (green arrow pointing DOWN): → Next decision

**DECISION 4 - Sparsity Check (WARNING):**
- Diamond shape with amber border
- Label inside: "M ≥ n·log(n)?"
- Check: "Sufficient comparison count?"
- TWO outputs:
  - NO (amber arrow pointing RIGHT): → "WARNING: Sparse Data" (amber rectangle)
    - Small annotation: "Results may have high variance"
    - Arrow continues DOWN (non-blocking)
  - YES (green arrow pointing DOWN): → Schema Inference

**PROCESS - Schema Inference (after all checks):**
- Large blue rounded rectangle
- Label: "Schema Inference"
- Sub-items listed:
  - "Infer Format (pointwise/pairwise/multiway)"
  - "Infer BigBetter direction"
  - "Extract Ranking Items"
  - "Detect Indicator Column"
- Arrow pointing DOWN to end

**END (bottom-center):**
- Green rounded rectangle
- Label: "Valid Schema Output"
- Sub-label: "Proceed to Configuration"

---

**VISUAL ANNOTATIONS:**

**Severity Legend (top-right corner):**
- Small box with three rows:
  - Red circle + "Critical: Blocks execution"
  - Amber circle + "Warning: Proceed with caution"
  - Green circle + "Valid: Full analysis available"

**Flow Annotations:**
- Use solid arrows for main flow (gray or black)
- Red arrows for rejection paths
- Amber arrows for warning paths
- Green arrows for success paths
- Label each decision with the validation rule

**KEY VISUAL ELEMENTS TO EMPHASIZE:**
1. Clear hierarchical structure: Critical checks first, then warnings
2. Non-blocking nature of warnings (flow continues despite warnings)
3. Terminal states for critical failures (red boxes)
4. Warning annotations explain impact on analysis
5. Final schema inference only reached after passing critical checks

**TYPOGRAPHY:**
- Decision labels: Bold, 11pt
- Process labels: Regular, 10pt
- Annotations: Italic, 9pt
- Mathematical notation: M, n, log(n) in standard math font

**OVERALL AESTHETIC:**
- Clean, minimal, professional scientific flowchart
- Similar to validation flowcharts in software engineering papers
- Landscape orientation (16:9 ratio) for horizontal flowchart layout
- Clear visual hierarchy showing validation severity levels
- Emphasis on the tiered feedback approach: errors block, warnings inform
