## Example Prompt 1

Create a detailed scientific architecture diagram illustrating the PennPRS Agent agentic AI system for automated statistical genetics literature curation and Polygenic Risk Score analysis. The diagram should be in a clean, professional scientific illustration style similar to figures in Nature Genetics papers, using simple geometric shapes, clean lines, and minimal icons.

**CRITICAL STYLE REQUIREMENTS:**
- Use only simple geometric shapes (rectangles, circles, arrows, lines)
- All icons should be abstract/schematic (like scientific flowchart symbols), NOT realistic illustrations
- Use flat design with no gradients, shadows, or 3D effects
- Text labels should be clean sans-serif font
- Overall style: hand-drawn scientific diagram aesthetic, similar to LaTeX/TikZ generated figures
- Include official logos for: PubMed (NCBI blue/white logo), PGS Catalog (their official logo with DNA helix), PennPRS (Penn Medicine style), BIGA (bigagwas.org branding)

**Aspect Ratio:** 3:4 (portrait orientation, suitable for journal figure)

**Color Scheme:**
- Deep navy blue (#011f5b, Penn Blue) for main system components and LLM Agent
- Light blue (#60a5fa) for data flow arrows
- Emerald green (#10b981) for PRS Performance module
- Amber/Gold (#f59e0b) for Heritability module
- Violet purple (#8b5cf6) for Genetic Correlation module
- Neutral gray (#9ca3af) for traditional pipeline comparison
- White background, no gradients

---

**LAYOUT STRUCTURE (Top to Bottom):**

The diagram is organized vertically in 5 horizontal bands:

**BAND 1 (Top): User Input & Disease Query**
**BAND 2: Core Innovation - LLM-Powered Literature Curation Engine**
**BAND 3: Three Sub-Modules (PRS Performance, Heritability, Genetic Correlation)**
**BAND 4: Cross-Module Integration & Unified Genetic Profile**
**BAND 5 (Bottom): Training APIs & External Data Sources**

---

**DETAILED PANEL DESCRIPTIONS:**

**Panel a - User Query Interface (top-center):**
- Draw a simple search bar with text "Alzheimer's Disease"
- User icon (simple circle with person silhouette) on the left
- Arrow labeled "Disease Query" pointing downward to the system
- Small label showing supported query types: "Disease name, Gene symbol, Trait ID"
- Text annotation: "Natural language disease/trait search"
- Label: "a. User query interface: Disease name input triggers comprehensive genetic profile retrieval"

**Panel b - LLM-Powered Literature Curation Engine (second band, full width):**
- Large central rectangle representing the "Unified Data Layer"
- Inside the rectangle, show three connected components:
  
  LEFT component: Literature Discovery
  - PubMed logo (official NCBI logo)
  - Arrow pointing to "Weekly automated scan"
  - Query examples in small text:
    - '"Alzheimer" AND "PRS"'
    - '"Alzheimer" AND "heritability"'  
    - '"Alzheimer" AND "genetic correlation"'
  
  CENTER component: LLM Agent (brain/gear icon)
  - Label: "LLM Agent"
  - Sub-labels: "Zero-shot extraction", "Structured parsing", "PMID linking"
  - Show extraction targets as small boxes: "AUC", "R²", "h²", "rg", "SE", "N"
  
  RIGHT component: Structured Database
  - Database cylinder icon
  - Label: "PennPRS Database"
  - Sub-label: "PGS-compatible schema"
  - Show sample fields: "trait, method, ancestry, pmid"

- Comparison callout box (gray, positioned to the side):
  - Title: "vs. Traditional Curation"
  - Row 1: "Manual review: Weeks → LLM extraction: Seconds"
  - Row 2: "PRS only → PRS + h² + rg"
  - Row 3: "Training required → Zero-shot"

- Label: "b. Core innovation: LLM-powered literature curation with automatic PubMed extraction and structured database construction"

**Panel c - Three Sub-Modules (third band, split into three equal columns):**

LEFT column - PRS Performance (green accent):
- Header box: "PRS Performance"
- Icon: Bar chart with trend line
- Key metrics displayed:
  - "Best AUC: 0.78"
  - "Best R²: 0.08"
  - "47 models found"
- Data sources (show logos):
  - PGS Catalog logo
  - "LLM-curated papers"
- Bottom action button: "Train Custom Model"
- Arrow pointing down to "PennPRS API"
- Label: "c1. PRS Performance: Model discovery from PGS Catalog + LLM-curated literature, with PennPRS training capability"

CENTER column - Heritability h² (amber/gold accent):
- Header box: "Heritability (h²)"
- Icon: Gauge/meter showing percentage
- Key metrics displayed:
  - "SOTA: h² = 0.24"
  - "SE: 0.03"
  - "N = 455,258"
  - "Method: LDSC"
- Source citation: "Jansen 2019 [PMID link]"
- Historical trend mini-chart showing h² estimates over years (2017-2024)
- Gap Analysis box:
  - "Efficiency = R²/h² = 33%"
  - "67% improvement potential"
- Bottom label: "Read-only (no train API)"
- Label: "c2. Heritability: SNP-heritability estimates with provenance, setting theoretical PRS ceiling and enabling gap analysis"

RIGHT column - Genetic Correlation (violet/purple accent):
- Header box: "Genetic Correlation (rg)"
- Icon: Network/connection diagram showing linked nodes
- Key correlations displayed as a mini table:
  - "T2D: rg = +0.38"
  - "Depression: rg = +0.42"
  - "Education: rg = -0.32"
- Data sources:
  - "LLM-curated papers"
- LLM Summary bubble:
  - "AD correlates with metabolic and psychiatric traits"
- Bottom action button: "Calculate Custom rg"
- Arrow pointing down to "BIGA API" with BIGA logo
- Label: "c3. Genetic Correlation: Cross-trait rg estimates with BIGA API training for custom correlation calculation"

**Panel d - Cross-Module Integration (fourth band, full width):**
- Central large box titled "Unified Genetic Profile: Alzheimer's Disease"
- Three columns inside showing summary from each module:
  
  Column 1 (Heritability):
  - "h² = 0.24"
  - "Ceiling for prediction"
  - "[PMID:30617256]"
  
  Column 2 (PRS):
  - "Best AUC = 0.78"
  - "Best R² = 0.08"
  - "[PMID:38xxxxxx]"
  
  Column 3 (rg):
  - "T2D: +0.38"
  - "Depression: +0.42"
  - "[Multiple PMIDs]"

- Below the three columns, a horizontal bar labeled "Cross-Module Insights (LLM-Generated)"
- Inside the bar, show LLM reasoning output:
  - "Current PRS captures 33% of h². Consider multi-trait PRS with T2D (rg=0.38) to improve prediction."
- Arrows showing data flow between modules

- Label: "d. Cross-module integration: Unified genetic profile with automated gap analysis and LLM-generated actionable insights"

**Panel e - External APIs & Data Sources (bottom band):**
- Four boxes arranged horizontally, each with official logo:

  Box 1: PubMed (NCBI)
  - Official PubMed logo (blue "PubMed" text with NCBI branding)
  - Label: "Literature Source"
  - Sub-label: "E-utilities API"
  - Description: "Weekly scan, ~30 papers/week"
  
  Box 2: PGS Catalog
  - Official PGS Catalog logo (DNA helix design)
  - Label: "PRS Reference"
  - Sub-label: "REST API"
  - Description: "5,000+ published models"
  
  Box 3: PennPRS
  - Penn Medicine style logo
  - Label: "PRS Training"
  - Sub-label: "Training API"
  - Description: "Custom model training"
  - Green action indicator
  
  Box 4: BIGA
  - BIGA logo (bigagwas.org style)
  - Label: "rg Training"
  - Sub-label: "Calculate rg"
  - Description: "LDSC, HDL, GNOVA methods"
  - Purple action indicator

- Dashed arrows connecting each box upward to relevant module
- Label: "e. External data sources and training APIs with official platform integrations"

---

**VISUAL FLOW ANNOTATIONS:**
- Use solid arrows (→) with light blue color for main data/workflow flow
- Use dashed arrows (-->) for optional paths or API calls
- Number each panel clearly with letters (a, b, c1, c2, c3, d, e) in navy circles
- Keep arrow styles consistent throughout
- Show bidirectional arrows between LLM Agent and external APIs

**KEY VISUAL ELEMENTS TO EMPHASIZE:**
1. The central role of LLM Agent in automated extraction
2. Every data point links to PMID (show small link icons)
3. Gap Analysis calculation: R²/h² ratio visualization
4. Contrast between "Read-only" (h²) vs "Trainable" (PRS, rg) modules
5. Official logos for external platforms (PubMed, PGS Catalog, PennPRS, BIGA)

**TYPOGRAPHY:**
- Panel labels: Bold, navy blue (#011f5b), positioned below each panel
- Component labels: Regular weight, black text
- Annotations: Smaller italic text for explanatory notes
- Metrics: Monospace font for numerical values (h²=0.24, rg=+0.38)

**OFFICIAL LOGOS TO INCLUDE:**
1. PubMed/NCBI: Blue "PubMed" wordmark with NCBI affiliation
2. PGS Catalog: Official logo with "PGS Catalog" text and DNA motif
3. PennPRS: "PennPRS" in Penn Medicine brand colors (Penn Blue #011f5b)
4. BIGA: "BIGA" text logo from bigagwas.org

**OVERALL AESTHETIC:**
- Clean, minimal, professional scientific figure style
- Similar to TikZ/LaTeX-generated diagrams in Nature Genetics
- No photorealistic elements or AI-generated artifacts
- Suitable for Nature Genetics or similar top-tier genetics journal
- Portrait orientation (3:4 ratio) optimized for single-column print figure
- Clear visual hierarchy: User → LLM Engine → Three Modules → Integration → APIs

**ALZHEIMER'S DISEASE EXAMPLE DATA TO SHOW:**
- PRS: AUC=0.78, R²=0.08, 47 models, PRS-CS method
- Heritability: h²=0.24 (SE=0.03), N=455,258, LDSC, Jansen 2019
- Genetic Correlations: T2D (+0.38), Depression (+0.42), Education (-0.32), CAD (+0.25)
- Gap Analysis: Efficiency = 0.08/0.24 = 33%, 67% improvement potential


## Example Prompt 2

Prompt for Visualization
Create a detailed scientific workflow diagram illustrating the PGS Catalog Data Collection & Processing Pipeline for Polygenic Score (PGS) curation. The diagram should be in a clean, professional bioinformatics illustration style, like a data pipeline flowchart from a Nature Methods paper, with labeled arrows showing data flow and processing relationships.

Key elements to include:
1. Overall Structure:

Horizontal timeline-based flowchart with 8 sequential stages, flowing left to right.
Use a modern, flat design aesthetic with a light gray or white background.
Color-coded sections: Blue tones for data input stages, Orange/Yellow for human-involved curation, Green for computational processing, Purple for output/distribution.
2. Step 1 - Literature Discovery (Left Section, Blue):

Depict a large PubMed database icon (stylized stack of papers or DNA helix with documents).
Show a "LitSuggest ML Classifier" box with a neural network icon inside, connected to PubMed via an arrow labeled "Weekly Automated Scan".
Output: A ranked list of candidate publications with probability scores (e.g., "PMID:38xxxxxx, Score: 0.92").
Include a small human curator icon reviewing the list with a checkmark/cross symbol, indicating "Triage Decision".
Triage categories shown as branching labels: "New PGS" (green check), "Evaluated PGS" (blue), "Not PGS" (red X, feeds back as negative training sample).
Add a feedback loop arrow from "Not PGS" back to "LitSuggest ML" labeled "Retraining Data".
3. Step 2 - Curation & Import (Central-Left Section, Orange):

Central focus: A stylized Excel spreadsheet icon with multiple tabs visible ("Publication", "Score", "Sample", "Cohort", "Performance").
Show a human curator icon sitting at a desk, reading a PDF/publication document on one side, and filling the Excel template on the other.
Extracted fields explicitly labeled with small text boxes: "Trait (EFO ID)", "AUC: 0.78", "R²: 0.08", "Sample Size: 388,000", "Ancestry: European", "Method: PRS-CS", "Variants: 84".
Arrow from spreadsheet to a "template_parser.py" processing box (Python logo + script icon).
Arrow from Python script to a PostgreSQL database cylinder icon (labeled "Django ORM → PGS Catalog DB").
Secondary input: Show an "EuropePMC API" cloud icon providing author names, journal, DOI metadata flowing into the spreadsheet.
Scoring file preparation: Depict a separate file icon (txt.gz format) with columns visible: "rsID | chr | pos | effect_allele | weight", labeled "Scoring File (Variant Weights)".
4. Step 3 - Validation & QC (Compact Middle Box, Yellow):

Small box with a magnifying glass + checkmark icon.
Arrows pointing in from "Scoring File" and out to next stage.
Labels inside: "Schema Validation", "Ensembl API Check", "Match Rate ≥ 90%".
Show a small warning icon for files that fail QC, with a red arrow looping back to "Curation" stage.
5. Step 4 - Harmonization / Liftover (Central-Right Section, Green, Detailed):

This section should be expanded with more detail than others.
Input: Original scoring file icon (labeled "Author-Reported Build: GRCh37").
Process Box 1 - HmPOS (Harmonize Position):
If rsID only: Arrow to "pgs_variants_coords" knowledge base (database cylinder icon) or "Ensembl Variation API" (cloud icon), returning chr:pos coordinates.
If coordinates exist: Arrow to "pyliftover" tool icon (chain/DNA helix transforming), labeled "GRCh37 → GRCh38 Coordinate Conversion".
Process Box 2 - HmVCF (Harmonize VCF):
Arrow from HmPOS output to "Ensembl Reference VCF" (large file icon with VCF label).
Show variant allele comparison: Depict a small table with columns "Effect Allele | Other Allele | Ref VCF Allele".
Strand flip correction: Icon showing A↔T, C↔G reversal with "Strand Flip" label.
Add "other_allele" completion: Arrow adding missing allele information.
Output: Three file icons emerging:
Original file (gray, labeled "Author-Reported").
Harmonized GRCh37 file (blue, labeled "Harmonized GRCh37").
Harmonized GRCh38 file (green, labeled "Harmonized GRCh38").
Harmonization Code Legend (Small Table):
hm_code 5: Green checkmark, "Mapped to Reference VCF".
hm_code 0: Yellow warning, "Author-Reported Only".
hm_code -4: Orange arrow, "Strand Flip Corrected".
hm_code -5: Red X, "Not Found in Reference".
6. Step 5 - Release & FTP (Right Section, Purple):

Show a release version tag icon (e.g., "v2024-01-15").
Arrow to an FTP server icon (cloud with download arrow, labeled "ftp.ebi.ac.uk/pub/databases/spot/pgs/").
Bulk metadata export: CSV/JSON file icons.
7. Step 6-8 - API & User Access (Far Right, Gradient Purple to Teal):

REST API: Django REST Framework logo + API endpoint examples ("/rest/score/{pgs_id}").
Website: Browser window icon showing pgscatalog.org search interface.
Score Calculation: Nextflow workflow icon (pgsc_calc logo) with input genotype files (PLINK/VCF icons) and output score report icon.
End user: Stylized researcher icon receiving the final PGS report.
8. Before-and-After Comparison (Bottom Strip, Optional):

Split into two horizontal bands:
Top Band - "Manual Curation Bottleneck": Red/orange tones, human icons overwhelmed by stacks of papers, clock icon showing slow processing.
Bottom Band - "Automated Pipeline": Green/blue tones, streamlined arrows, fast clock icon, showing efficiency gains from computational processing.
9. Labels and Annotations:

Use clear, sans-serif text labels for all components (e.g., "LitSuggest ML", "Excel Template", "pgs-harmonizer", "pgsc_calc").
Arrows indicating data flow with verb labels (e.g., "Parses", "Validates", "Lifts Over", "Stores", "Serves").
Inhibition/blocking arrows: Use red X icons where data fails QC or is rejected.
Legend box in bottom-right corner explaining color codes and symbols.
10. Overall Aesthetic:

High-resolution, vector-style graphics suitable for publication or presentation.
Primary color palette: Deep blue (#1E3A5F) for input, Orange (#E07020) for human curation, Green (#28A745) for computation, Purple (#6F42C1) for output.
Minimal background, no gradients or textures that distract from data flow.
Icons should be simple, recognizable (flat design, similar to FontAwesome or Material Design icons).
Suitable for educational slides, grant proposals, or pharmaceutical/biotech research reports.
Aspect ratio: Wide (16:9) for presentation slides, or square (1:1) for journal figures.