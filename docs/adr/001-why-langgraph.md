# ADR 001 — Why LangGraph over a simple LangChain chain

**Date:** Week 1  
**Status:** Accepted

## Context
DocuSage needs to route queries through different agent paths depending on whether the query involves visual pages (tables/charts), pure text, or both. A linear chain cannot express conditional branching.

## Decision
Use **LangGraph** to build a stateful, branching agent graph.

## Rationale
| Option | Problem |
|---|---|
| Simple LangChain chain | Linear only — cannot skip VLM extraction for text queries |
| LangChain AgentExecutor | Tool-calling loop — unpredictable execution path, hard to test |
| Custom Python routing | Works, but loses observability, streaming, and checkpointing |
| **LangGraph** | Explicit graph definition, conditional edges, built-in state management, Langfuse traces each node |

## Consequences
- The graph definition (`agents/graph.py`) is the single source of truth for execution flow
- Each agent is a pure function: `DocuSageState → DocuSageState` — easy to unit test
- Adding new agents (e.g., a summarization agent) = add a node + edge, no refactoring
