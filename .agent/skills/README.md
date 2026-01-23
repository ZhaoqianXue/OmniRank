# Antigravity Agent Skills

This directory contains the core intelligence modules and specialized capabilities for the Antigravity coding assistant. Each subdirectory represents a standalone "Skill" documented via a SKILL.md file.

## Quick Statistics
- Total Skills Installed: 25
- Management System: Modified skills-updater (compatible with local .agent/skills logic)

---

## Skill Categories

### Scientific and Academic Research
Tools designed for high-end academic workflows and scientific discovery.
- academic-writing-citation: Search PubMed/Google Scholar, manage BibTeX, and validate citations.
- scientific-visualization: Create publication-quality (Nature/Science/Cell style) figures using Matplotlib/Seaborn.
- plotly: Build interactive dashboards and exploratory data visualizations.

### Document Intelligence
Comprehensive processing for professional document formats.
- docx-skill: Microsoft Word document creation, editing, and tracked changes.
- pdf-skill: Text/Table extraction, form filling, merging, and splitting of PDFs.
- pptx-skill: Automated PowerPoint generation with professional layouts.
- xlsx-skill: Advanced Excel data analysis, formulas, and spreadsheet automation.
- markdown-tools: High-fidelity conversion of documents (PDF, Docx, PPTX) to clean Markdown.

### Superpowers (Advanced Workflows)
Specialized sub-agents and logic for production-grade software engineering.
- systematic-debugging: Step-by-step root cause analysis and bug fixing.
- subagent-driven-development: Dispatches independent tasks to parallel sub-agents.
- writing-plans and executing-plans: Systematic implementation of complex multi-step features.
- test-driven-development (TDD): Ensuring code reliability through test-first methodologies.
- using-git-worktrees: Isolated workspace management for safe feature development.
- verification-before-completion: Automated check and verification before finalizing tasks.
- brainstorming: Intent exploration and architectural design before execution.
- requesting-code-review and receiving-code-review: Rigorous peer-review simulation.

### Management and Utility
- skills-updater: [Primary Management Tool] Scans this folder for updates, recommends new skills, and handles local merges.
- planning-with-files: Persistent research and session state management using project-local markdown files.
- frontend-design: High-end, distinctive UI/UX component generation.

---

## How to Use
These skills are automatically detected by Antigravity. You can invoke them via:
1.  Natural Language: "Create a publication-quality plot for this data" -> Triggers scientific-visualization.
2.  Explicit Command: "/skills-updater check" -> Triggers the update check logic.
3.  Proactive Assistance: Antigravity will automatically load relevant SKILL.md instructions when tackling complex tasks (like deep research or UI design).

## Management (skills-updater)
To maintain the health of your local skill library, use the dedicated tool:
```bash
# Check for updates in the local .agent/skills directory
python3 .agent/skills/skills-updater/scripts/check_updates.py --lang zh

# Get new skill recommendations
python3 .agent/skills/skills-updater/scripts/recommend_skills.py --lang zh
```

---
*Maintained by Antigravity Assistant.*
