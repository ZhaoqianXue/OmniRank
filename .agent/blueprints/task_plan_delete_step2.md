# Task Plan: Delete Engine Orchestrator Step 2 Content

## Objective
Remove all Step 2 related content from the OmniRank project while retaining only Step 1 functionality.

## Status Summary
| Phase | Status | Notes |
|-------|--------|-------|
| 1. Document Analysis | complete | Identified all files with Step 2 references |
| 2. SOP Document Updates | in_progress | Main architecture document |
| 3. Writing Document Updates | pending | Paper writing document |
| 4. Backend Code Updates | pending | Python orchestrator, r_executor, etc. |
| 5. Frontend Code Updates | pending | TypeScript/React components |
| 6. Figure Prompts Updates | pending | docs/figure/ directory |
| 7. Experiment Cleanup | pending | experiments/orchestrator_step2/ |
| 8. R Script Cleanup | pending | spectral_ranking_step2.R deletion |
| 9. Verification | pending | Final consistency check |

## Files to Modify

### Documents (Priority 1)
1. `.agent/blueprints/sop.md`
   - Lines 152-160: Multi-Agent Collaboration Workflow - Step 2 references
   - Lines 203-204: Self-Correcting Mechanism - Step 2 references
   - Lines 303-314: Spectral Ranking Pipeline - STEP 2a/2b
   - Lines 378: Agent Roles Table - spectral_ranking_step2.R mention
   - Lines 427-435: Workflow Algorithm - should_refine and execute_step2 logic
   - Lines 468-469: Workflow Phases Table - Step 2 references
   - Lines 490-502: Tool Ecosystem - ranking_engine_step2 definition
   - Lines 671-735: Engine Orchestrator Agent - Two-Step Decision Logic
   - Lines 779-812: R Script descriptions for Step 2
   - Lines 979-1009: Optimal Weighting/Two-Step section
   - Lines 1083-1087: Workflow Example - Step 2 triggering
   - Lines 1234-1238: Algorithm pseudocode line 6-10
   - Lines 1277-1284: Engine Orchestrator Function 2 - Two-Step Method

2. `.agent/blueprints/writing.md`
   - Section 3.3.2 Adaptive Two-Step Spectral Estimation (lines 164-178)
   - Section 4.2.3 Two-Step Method Triggering (lines 375-411)
   - Algorithm 1 lines 91-95 (Phase 3 Step 2 logic)
   - Various references to two-step method

### Backend Code (Priority 2)
1. `src/api/agents/orchestrator.py`
   - Step2 decision logic
   - should_run_step2 imports/usage
   - _decide_step2 method
   - execute_step2 calls

2. `src/api/core/r_executor.py`
   - Step2Params, Step2Result classes
   - STEP2_SCRIPT constant
   - run_step2 method
   - should_run_step2 function

3. `src/api/core/__init__.py`
   - Step2 exports

4. `src/api/agents/analyst_agent.py`
   - step2_triggered references

5. `src/api/api/routes.py`
   - Step 2 execution comments

### Frontend Code (Priority 3)
1. `src/web/lib/api.ts`
   - step2_triggered type field

2. `src/web/components/chat/chat-input.tsx`
   - step2_triggered references

3. `src/web/components/report/report-overlay.tsx`
   - step2_triggered display logic

### Figure Prompts (Priority 4)
1. `docs/figure/figure1_prompt.md`
2. `docs/figure/figure3_prompt.md` (entire file - Two-Step Decision Tree)
3. `docs/figure/figure5_prompt.md`
4. `docs/figure/figure6_prompt.md`

### Experiments (Priority 5)
1. `experiments/orchestrator_step2/` - Entire directory to be archived/deleted

### R Scripts (Priority 6)
1. `src/spectral_ranking/spectral_ranking_step2.R` - Delete

## Key Modifications Summary

### Engine Orchestrator Changes
- Remove: Decision logic for Step 2   
- Remove: execute_step2() method calls
- Remove: should_run_step2() function usage
- Simplify: Execute only Step 1 (spectral_ranking.R)

### Architecture Description Changes
- Remove: "Two-Step Spectral Method" descriptions
- Remove: Heterogeneity/Uncertainty triggers for Step 2
- Remove: Optimal weighting discussions
- Retain: Step 1 vanilla spectral ranking with uniform weights

### Paper Writing Changes
- Remove: Section 4.2.3 Two-Step Method Triggering
- Remove: Algorithm 1 lines 9-12 (Step 2 conditional logic)
- Simplify: Section 3.3.2 to only mention Step 1

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| (none yet) | - | - |
