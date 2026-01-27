# Figure 6 Prompt: OmniRank User Interface

Create a detailed scientific diagram illustrating the OmniRank web-based user interface with its three-stage workflow. The diagram should be in a clean, professional scientific illustration style similar to figures in Nature Methods papers, showing UI mockups/wireframes for each stage.

**CRITICAL STYLE REQUIREMENTS:**
- Use wireframe/mockup style for UI elements (not screenshots)
- Simple geometric shapes representing UI components
- Use flat design with no gradients, shadows, or 3D effects
- Text labels should be clean sans-serif font
- Overall style: scientific software interface diagram, similar to user study figures in CHI papers
- DO NOT include any figure title or caption in the diagram itself

**Aspect Ratio:** 3:4 (portrait orientation, suitable for journal figure)

**Color Scheme:**
- Deep blue (#1E40AF) for primary UI elements and headers
- Emerald green (#059669) for success states and action buttons
- Amber/Gold (#D97706) for warning indicators
- Light gray (#E5E7EB) for UI containers and backgrounds
- Light purple (#7C3AED) for interactive elements
- White (#FFFFFF) for content areas

---

**LAYOUT STRUCTURE:**

The diagram is organized as THREE horizontal panels (a, b, c) stacked vertically, each showing one stage of the workflow:

**Panel (a): Stage 1 - Data Upload and Schema Inference**
**Panel (b): Stage 2 - Interactive Configuration**
**Panel (c): Stage 3 - Results and Exploration**

Each panel shows a simplified wireframe of the interface at that stage.

---

**DETAILED PANEL DESCRIPTIONS:**

**Panel (a) - Stage 1: Data Upload and Schema Inference:**
- Panel label: "a" in blue circle (top-left)
- Stage indicator: "Stage 1: Data Upload"

LEFT side - Upload Area:
- Large dashed rectangle (drop zone)
- Cloud upload icon in center
- Text: "Drop CSV/Excel file here"
- "or click to browse"
- File format badges: "CSV", "Excel"
- Example filename shown: "llm_benchmark_scores.csv"

RIGHT side - Schema Display:
- Card titled "Inferred Schema"
- Content rows:
  - "Format: Pointwise" (with green checkmark)
  - "Items: 8 models detected"
  - "Indicator: Task (5 categories)"
  - "Direction: Higher is better"
- Validation status section:
  - Green checkmark: "Data integrity ✓"
  - Green checkmark: "Connectivity ✓"
  - Amber warning: "⚠ Sparsity: borderline"

BOTTOM - Data Preview:
- Small table preview (3 rows × 4 columns)
- Column headers: "Task", "GPT-4", "Claude", "Gemini"
- Sample values shown
- "Preview: first 3 rows of 50"

**Panel (b) - Stage 2: Interactive Configuration:**
- Panel label: "b" in blue circle (top-left)
- Stage indicator: "Stage 2: Configuration"

LEFT side - Parameter Controls:
- Card titled "Analysis Settings"
- Control groups:

  Group 1 - Direction:
  - Radio buttons: "● Higher is better" / "○ Lower is better"
  - Small info icon with tooltip hint
  
  Group 2 - Item Selection:
  - Checkbox list:
    - "☑ GPT-4"
    - "☑ Claude-3"
    - "☑ Gemini-Pro"
    - "☑ Llama-3"
    - "☐ GPT-3.5 (exclude)"
  - "Select all" / "Clear" links
  
  Group 3 - Indicator Filter:
  - Dropdown or chip selector
  - Selected: "Code", "Math", "Writing"
  - Unselected: "Translation", "QA"

RIGHT side - Advanced Options (collapsed by default):
- Expandable section titled "Advanced Settings"
- Bootstrap iterations: "2000" (input field)
- Random seed: "42" (input field)
- Small text: "For reproducibility"

BOTTOM - Action Buttons:
- Primary button (green): "Run Analysis"
- Secondary button (gray): "Reset to Defaults"
- Loading indicator placeholder

**Panel (c) - Stage 3: Results and Exploration:**
- Panel label: "c" in blue circle (top-left)
- Stage indicator: "Stage 3: Results"

TOP section - Summary Cards:
- Three small metric cards in a row:
  - "#1: GPT-4" (with trophy icon)
  - "8 items ranked"
  - "Step 2 applied ✓"

LEFT side - Visualization Panel:
- Tabbed interface: "Forest Plot" | "Heatmap" | "Distribution"
- Forest plot wireframe shown:
  - Vertical axis: item names (GPT-4, Claude, Gemini...)
  - Horizontal axis: rank with error bars
  - Point estimates with CI whiskers
  - Items sorted by rank

RIGHT side - Report Panel:
- Scrollable text area
- Section headers visible:
  - "## Executive Summary"
  - "## Statistical Significance"
  - "## Recommendations"
- Partial text content indicated with wavy lines
- Export buttons: "PDF", "PNG"

BOTTOM section - Chat Interface:
- Chat input bar with placeholder: "Ask a follow-up question..."
- Send button (purple)
- Suggested questions as clickable chips:
  - "Is GPT-4 significantly better than Claude?"
  - "Which rankings have high uncertainty?"
  - "Why was Step 2 applied?"
- Previous Q&A exchange shown:
  - User: "Are there statistical ties?"
  - Agent: "Yes, Claude-3 and Gemini-Pro have overlapping CIs..."

---

**VISUAL ANNOTATIONS:**

**Workflow Arrow (connecting panels):**
- Vertical arrow on the left side connecting all three panels
- Labels at each transition:
  - "Upload" → "Inference complete"
  - "Configure" → "Analysis running"
  - "Results" → "Interactive exploration"

**Stage Progress Indicator (top of each panel):**
- Three-step progress bar: ●——●——○ (filled/active/pending)
- Current stage highlighted

**KEY VISUAL ELEMENTS TO EMPHASIZE:**
1. Progressive disclosure: complexity increases from (a) to (c)
2. Human-in-the-loop: user confirmation required between stages
3. Interactive elements: checkboxes, dropdowns, chat input
4. Dual output: visualizations + natural language report
5. Suggested questions: guiding user exploration

**TYPOGRAPHY:**
- Panel labels: Bold, 14pt in blue circles
- Section headers: Bold, 11pt
- UI element labels: Regular, 10pt
- Placeholder text: Italic, gray, 9pt
- Metric values: Monospace, 10pt

**UI ELEMENT STYLING:**
- Buttons: Rounded rectangles with 4px radius
- Cards: Light gray background with subtle border
- Input fields: White with gray border
- Checkboxes/Radio: Simple geometric shapes
- Icons: Minimal line icons (not filled)

**OVERALL AESTHETIC:**
- Clean, minimal wireframe style
- Similar to user interface figures in HCI papers
- Portrait orientation (3:4 ratio) with three stacked panels
- Clear visual progression through workflow stages
- Emphasis on accessibility: no-code interface for domain experts
- Modern web application aesthetic (similar to Streamlit/Gradio apps)
