# OmniRank Todo List

## AboutLLM Agent
- [x] **Knowledge Layer**: Integrate literature (e.g., Mengxin Yu) for expert-level theoretical grounding in Agent prompts. - By using "Structured System Instructions" (OpenAI Recommended)
  - Implemented: `SPECTRAL_KNOWLEDGE` constant in `src/api/agents/analyst_agent.py`
  - Contains: Core concepts, CI interpretation, two-step method, heterogeneity, sparsity

## About spectral_ranking_inferences Paper - Optimal Weight Function Selection (Section 2.2, Theorem 2) - *Engine Orchestrator*
- [x] **Two-Step Spectral Method Implementation**: Implement the two-step spectral method where Step 1 uses simple weight $f(A_l) = |A_l|$ for initial consistent estimation, and Step 2 uses the estimated optimal weight $f(A_l) \propto \sum_{u \in A_l} e^{\theta^*_u}$ to achieve MLE-equivalent asymptotic efficiency.
- [x] **Automatic Two-Step Decision**: Agent can automatically decide whether to use the two-step method based on data characteristics (e.g., sample size, comparison graph density, heterogeneity level).
- [x] **Convergence Monitoring**: Agent can monitor the convergence of Step 1 estimation and determine whether Step 2 is necessary (e.g., if Step 1 already achieves acceptable precision, Step 2 may be optional).

## User Q&A (Follow-up Questions)
- [x] **Chat API Endpoint**: POST `/api/chat` for user questions
- [x] **Chat Input Component**: `src/web/components/chat/chat-input.tsx`
- [x] **Hook Integration**: `sendMessage` in `useOmniRank` hook

## Optional Enhancements
- [ ] **Export Functionality**: Export reports as MD/PDF, visualizations as PNG/PDF
- [ ] **Score Distribution Visualization**: Bootstrap distribution chart (4th tab)

## Error Diagnosis (Low Priority - Time Permitting)
- [ ] **Methodology**: Write Section 3.4 Error Diagnosis details (ReAct loop, DATA_ERROR vs EXECUTION_ERROR classification)
- [ ] **Experiments**: Design evaluation for Error Diagnosis success rate (inject common errors, measure recovery rate)
- [ ] **Coding**: Implement ReAct-based error diagnosis in Analyst Agent (`src/api/agents/analyst_agent.py`)
