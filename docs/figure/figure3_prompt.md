# Figure 3 Prompt: Engine Orchestrator Decision Tree for Two-Step Refinement

Create a detailed scientific decision tree illustrating the Engine Orchestrator's adaptive logic for determining whether to apply two-step spectral refinement. The diagram should be in a clean, professional scientific illustration style similar to figures in Nature Methods papers, using standard decision tree notation.

**CRITICAL STYLE REQUIREMENTS:**
- Use only standard flowchart symbols (rectangles for processes, diamonds for decisions)
- All icons should be abstract/schematic, NOT realistic illustrations
- Use flat design with no gradients, shadows, or 3D effects
- Text labels should be clean sans-serif font
- Overall style: hand-drawn scientific diagram aesthetic, similar to LaTeX/TikZ generated figures
- DO NOT include any figure title or caption in the diagram itself

**Aspect Ratio:** 3:4 (portrait orientation, suitable for journal figure)

**Color Scheme:**
- Emerald green (#059669) for the main orchestrator container
- Deep blue (#1E40AF) for Step 2 execution path
- Light gray (#E5E7EB) for Step 1 only path
- Amber/Gold (#D97706) for decision diamonds
- Coral red (#DC2626) for gatekeeper block
- White background, no gradients

---

**LAYOUT STRUCTURE (Top to Bottom):**

The decision tree follows a hierarchical structure:

**LEVEL 1: Step 1 Execution (always runs)**
**LEVEL 2: Gatekeeper Check (blocks if fails)**
**LEVEL 3: Trigger Evaluation (activates Step 2 if any trigger fires)**
**LEVEL 4: Final Output**

---

**DETAILED DECISION TREE ELEMENTS:**

**INPUT (top-center):**
- Rectangle with dashed border
- Label: "From Data Agent"
- Sub-label: "Schema + Data Path"
- Arrow pointing DOWN

**PROCESS 1 - Step 1 Execution (always runs):**
- Large green rectangle
- Label: "Step 1: Vanilla Spectral Estimation"
- Inside box, show:
  - R logo icon (small)
  - Formula: "f(Aₗ) = |Aₗ|" (uniform weighting)
  - "Bootstrap: 2000 iterations"
- Output annotation: "Produces: θ̂⁽¹⁾, Metadata"
- Arrow pointing DOWN labeled "Metadata"

**GATEKEEPER DECISION (critical gate):**
- Diamond shape with RED border (emphasized as blocking)
- Label inside: "Sparsity Ratio ≥ 1.0?"
- Formula annotation below: "M / (n · log n) ≥ 1.0"
- TWO outputs:
  - NO (red arrow pointing RIGHT): → "BLOCK Step 2" (red rectangle)
    - Annotation: "Data too sparse for reliable refinement"
    - Arrow curves DOWN to final output (gray path)
  - YES (green arrow pointing DOWN): → Trigger Evaluation

**TRIGGER EVALUATION BOX:**
- Large amber/gold bordered rectangle containing TWO parallel decision diamonds
- Header: "Trigger Conditions (OR logic)"

  **TRIGGER A - Heterogeneity:**
  - Diamond shape
  - Label: "Heterogeneity > 0.5?"
  - Formula: "CV(comparison counts) > 0.5"
  - Annotation: "High variation in hyperedge sizes"
  
  **TRIGGER B - Uncertainty:**
  - Diamond shape
  - Label: "CI Width / n > 20%?"
  - Formula: "mean_ci_width / n_items > 0.2"
  - Annotation: "High estimation variance"

- Logic connector: "OR" symbol between the two triggers
- Combined output:
  - ANY YES (blue arrow pointing DOWN): → Step 2 Execution
  - ALL NO (gray arrow pointing RIGHT): → Skip to final output

**PROCESS 2 - Step 2 Execution (conditional):**
- Large blue rectangle (highlighted as optional/conditional)
- Label: "Step 2: Optimal Weight Refinement"
- Inside box, show:
  - R logo icon (small)
  - Formula: "f(Aₗ) ∝ Σᵤ∈Aₗ exp(θ̂ᵤ⁽¹⁾)"
  - "Achieves Cramér-Rao bound"
- Output annotation: "Produces: θ̂⁽²⁾ (refined)"
- Arrow pointing DOWN

**MERGE POINT:**
- Small circle where two paths converge:
  - Blue path from Step 2
  - Gray path from "Step 1 only" (bypassing Step 2)

**OUTPUT (bottom-center):**
- Rectangle with green border
- Label: "Ranking Results + Confidence Intervals"
- Sub-labels showing two possible sources:
  - "θ̂⁽¹⁾ (if Step 1 only)" in gray
  - "θ̂⁽²⁾ (if refined)" in blue
- Arrow pointing DOWN: "To Analyst Agent"

---

**VISUAL ANNOTATIONS:**

**Decision Summary Table (side panel, right):**
Small table showing the decision logic:
```
| Condition              | Threshold | Action if True      |
|------------------------|-----------|---------------------|
| Sparsity Ratio < 1.0   | Gatekeeper| BLOCK Step 2        |
| Heterogeneity > 0.5    | Trigger A | RUN Step 2          |
| CI Width / n > 20%     | Trigger B | RUN Step 2          |
```

**Path Labels:**
- Gray path: "Step 1 Only (data insufficient or balanced)"
- Blue path: "Step 2 Applied (data sufficient AND triggers met)"
- Red path: "Blocked (sparse data protection)"

**KEY VISUAL ELEMENTS TO EMPHASIZE:**
1. The GATEKEEPER as a hard block (red, prominent)
2. The OR logic between triggers (both paths lead to Step 2)
3. The optional nature of Step 2 (blue path is conditional)
4. Two possible outputs converging to same destination
5. Mathematical formulas for each decision threshold

**TYPOGRAPHY:**
- Decision labels: Bold, 11pt
- Process labels: Bold, 12pt
- Formulas: Math font (LaTeX style), 10pt
- Annotations: Italic, 9pt
- Thresholds: Monospace font for numbers (1.0, 0.5, 20%)

**OVERALL AESTHETIC:**
- Clean, minimal, professional scientific decision tree
- Similar to algorithm flowcharts in machine learning papers
- Portrait orientation (3:4 ratio)
- Clear visual distinction between mandatory (Step 1) and conditional (Step 2) paths
- Emphasis on the adaptive triggering logic based on data characteristics
