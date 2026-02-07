# Knowledge Base

This directory contains curated technical knowledge, engineering best practices, and research summaries relevant to the development and optimization of AI agents.

## Directory Structure

### [Context Engineering](./context_engineering/)
Strategies for managing and optimizing the information (tokens) passed to LLMs to improve agent performance, latency, and cost.

- **[Manus: Context Engineering](./context_engineering/manus_context_engineering.md)**: Lessons from building Manus, focusing on KV-cache optimization, tool masking, and using the file system as external memory.
- **[Anthropic: Effective Context Engineering](./context_engineering/anthropic_context_engineering.md)**: Anthropic's mental model for curating context, including the "Goldilocks Zone" for prompts and Just-in-Time (JIT) retrieval.
- **[Anthropic: Long-Running Agents](./context_engineering/anthropic_long_running_agents.md)**: Strategies for maintaining coherence across multiple context windows using Initializer/Coding agent roles and structured progress tracking.

### [Multi-Agent Systems](./multi_agent_system/)
Architectures, patterns, and best practices for building systems with multiple coordinated AI agents.

- **[Building Multi-Agent Systems: When and How to Use Them](./multi_agent_system/building_multi_agent_systems.md)**: Decision framework for multi-agent architectures, covering when to use them (context protection, parallelization, specialization) and when not to. Includes context-centric decomposition principles and the verification subagent pattern.
- **[How We Built Our Multi-Agent Research System](./multi_agent_system/how_we_built_multi_agent_research_system.md)**: Anthropic's engineering journey building their Research feature, including 8 prompt engineering principles, evaluation strategies, and production reliability challenges. Covers orchestrator-worker patterns, parallel tool calling, and long-horizon conversation management.

## How to use this knowledge

- **Context Engineering**: Reference these documents when designing prompt structures, tool definitions, or handling long-range dependencies in agent workflows.

- **Multi-Agent Systems**: 
  - Use the decision framework to determine if multi-agent architecture is appropriate for your use case
  - Reference prompt engineering principles when designing orchestrator and subagent prompts
  - Apply context-centric decomposition when dividing work between agents
  - Consider production reliability patterns when deploying multi-agent systems at scale
