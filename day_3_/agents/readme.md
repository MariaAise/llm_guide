---
# Day 3 — Agentic Systems: Architecture, Evaluation, and Design Foundations
---


## Contents

<!-- START doctoc -->
<!-- END doctoc -->

-------

## Learning Objectives

After engaging with this document, the reader should be able to:

1. Precisely define what constitutes an AI agent, distinguishing it from traditional predictive or generative AI systems.
2. Understand the core architectural components of agentic systems and how they interact in production settings.
3. Evaluate when and why agentic systems are appropriate compared to deterministic workflows or single-call LLM applications.
4. Reason about agent design trade-offs, including autonomy, security, observability, and scalability.
5. Apply principled evaluation and operational practices (Agent Ops) to ensure reliability, quality, and governance.
6. Identify inherent limitations of current agentic approaches and understand credible extension paths.

---

## Overview

Agentic systems represent a structural shift in how AI software is built and deployed. Rather than treating large language models (LLMs) as passive components that respond to isolated prompts, agentic systems place models inside a continuous control loop that allows them to reason, act, observe outcomes, and iteratively progress toward a goal.

An AI agent is best understood not as a model, but as a **goal-oriented application** composed of four tightly integrated elements:

* A reasoning model (the cognitive core)
* Tools (interfaces to data and actions)
* An orchestration layer (control logic and memory)
* Deployment and operational services (runtime, monitoring, security)

This framing moves AI from *assistive generation* toward *autonomous problem-solving*. However, autonomy introduces non-determinism, new failure modes, and governance challenges that require fundamentally different design and evaluation practices than traditional software or ML systems.

---

## Why Use Agentic Systems

Agentic systems are justified when problems exhibit the following characteristics:

1. **Multi-step structure**
   Tasks cannot be solved reliably in a single model invocation and require sequencing, planning, or conditional branching.

2. **Dynamic context**
   The information needed to solve the task is not fully available upfront and must be retrieved, computed, or updated during execution.

3. **Tool interaction**
   The system must query databases, call APIs, execute code, or interact with external systems rather than merely generate text.

4. **Adaptation under uncertainty**
   The correct next action depends on intermediate results, user feedback, or environmental changes.

Examples include research synthesis, operational coordination, compliance workflows, customer support resolution, and complex decision support.

Conversely, agentic approaches are *not* appropriate where:

* The task is fully deterministic and well-specified.
* Latency or cost constraints prohibit iterative reasoning.
* Regulatory or safety requirements demand strictly bounded behavior without runtime discretion.

---

## Architecture

At a high level, an agentic system operates as a closed loop:

**Think → Act → Observe → Update Context → Think**

This loop is implemented through the interaction of four architectural layers.

### 1. Model (Reasoning Core)

The model serves as the agent’s reasoning engine. Its responsibilities include:

* Interpreting goals
* Decomposing tasks
* Selecting tools
* Synthesising outputs

Model choice is an architectural decision, not a benchmark contest. Effective agentic models must demonstrate:

* Multi-step reasoning reliability
* Consistent tool invocation
* Stable behavior under context variation

In practice, production systems often employ **model routing**, using more capable models for planning and cheaper models for routine subtasks.

### 2. Tools (Action Interfaces)

Tools connect reasoning to reality. They fall into three primary categories:

* **Information retrieval** (RAG, search, databases, knowledge graphs)
* **Action execution** (APIs, code execution, system updates)
* **Human interaction** (confirmation, clarification, approval)

Tools are not optional add-ons; they are first-class components that define what the agent can and cannot do.

### 3. Orchestration Layer (Control Plane)

The orchestration layer governs:

* The execution loop
* State and memory management
* Reasoning strategies
* Tool invocation policies

It determines when the model reasons, when it acts, and how observations are incorporated into future context. This layer encodes design choices about autonomy, safety, and determinism.

### 4. Deployment and Runtime Services

Production agents require:

* Persistent memory
* Session management
* Logging and tracing
* Authentication and authorization
* Scalable infrastructure

Without these services, agents remain prototypes rather than reliable systems.

---

## Core Concepts

### Agentic Problem-Solving Loop

An agent executes a recurring cycle:

1. **Receive a mission** (user request or system trigger)
2. **Scan context** (memory, tools, environment)
3. **Reason and plan**
4. **Execute an action**
5. **Observe outcomes**
6. **Update context and iterate**

This loop continues until the mission is satisfied or terminated.

### Taxonomy of Agentic Systems

Agentic systems can be classified by capability:

* **Level 0 – Core Reasoning**
  Isolated LLM reasoning without tools or memory.

* **Level 1 – Connected Agent**
  Tool-enabled access to external data and APIs.

* **Level 2 – Strategic Agent**
  Multi-step planning with active context engineering.

* **Level 3 – Multi-Agent Systems**
  Coordinated teams of specialized agents.

* **Level 4 – Self-Evolving Systems**
  Agents that create or modify tools and sub-agents dynamically.

Each level introduces new power and new risks; higher levels are not strictly “better” and must be justified by the problem domain.

---

## Evaluation

Traditional pass/fail testing is insufficient for agentic systems. Evaluation must address **quality, reliability, and impact**.

### Metrics That Matter

Evaluation should be framed like an experiment:

* Task completion rate
* Latency
* Cost per task
* User satisfaction
* Business impact metrics

### Quality Evaluation with LM Judges

Because outputs are probabilistic, quality is assessed using rubric-based evaluation, often with an LLM acting as a judge. This allows scoring across dimensions such as correctness, grounding, compliance, and tone.

### Metrics-Driven Deployment

Changes to agents should be gated by comparative evaluation against a fixed test set. Deployment becomes a go/no-go decision informed by measured deltas, not intuition.

### Observability and Debugging

High-fidelity traces record:

* Prompts
* Model outputs
* Tool calls
* Parameters
* Observations

Tracing is essential for diagnosing failure modes that cannot be reproduced deterministically.

---

## Design Considerations

Key architectural decisions include:

* **Degree of autonomy**
  Fully autonomous vs tightly constrained workflows.

* **Context management**
  What is persisted, retrieved, or discarded.

* **Memory strategy**
  Short-term scratchpad vs long-term retrieval.

* **Multi-agent decomposition**
  Specialization vs monolithic agents.

* **Human-in-the-loop controls**
  Where confirmation or override is required.

* **Security posture**
  Identity, least-privilege access, and guardrails.

Agentic systems should be designed as **directed autonomy**, not open-ended intelligence.

---

## Limitations and Extensions

### Current Limitations

* Non-deterministic behavior complicates guarantees.
* Tool misuse can have real-world consequences.
* Evaluation is expensive and requires domain expertise.
* Long-running agents introduce state management complexity.
* Security risks increase with autonomy.

### Extension Paths

* Stronger agent identity and policy frameworks
* Improved simulation environments (“agent gyms”)
* Standardized agent-to-agent protocols
* Better economic primitives for agentic transactions
* Tighter integration of governance and observability

---

## Key Takeaway

Agentic systems are not simply “LLMs with tools.” They are a new class of software defined by continuous reasoning loops, context management, and autonomous action under uncertainty. Their power lies in solving problems that resist static workflows—but this power must be constrained, measured, and governed.

The success of an agentic system depends less on the brilliance of its prompts and more on the rigor of its architecture, evaluation discipline, and operational controls.

---

## Related Files

Below is a **revised, cleaned version of the “Related Files” section**.
All **dead, speculative, or non-public links have been removed or replaced**.
Only **verifiable, stable, publicly accessible sources** remain.
Where a concept is important but no stable public link exists, it is **explicitly marked as internal / conceptual** rather than linked.

---

## Related Files 

## 1. Foundational Agent Architecture & Taxonomy

Authoritative sources defining *what agents are*, how they differ from workflows, and how capability scales.

* **Introduction to Agents and Agent Architectures (Nov 2025)**
  Primary foundational document defining agent anatomy (Model, Tools, Orchestration, Deployment), agent taxonomy (Levels 0–4), and the agentic problem-solving loop.


* **ReAct: Synergizing Reasoning and Acting in Language Models**
  Foundational paper introducing the reasoning–action loop used in modern agent orchestration.
  [https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)

* **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**
  Early formalisation of structured reasoning relevant to agent planning.
  [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)

---

## 2. Agent Orchestration, Context & Memory

Sources focused on *control logic, context window management, and memory design*.

* **Agent Development Kit (ADK) Documentation**
  Code-first framework for building, orchestrating, and operating agents.
  [https://google.github.io/adk-docs/](https://google.github.io/adk-docs/)

* **Choosing a Design Pattern for Agentic AI Systems (Google Cloud)**
  Canonical reference for Coordinator, Sequential, Iterative Refinement, and HITL patterns.
  [https://cloud.google.com/architecture/choose-design-pattern-agentic-ai-system](https://cloud.google.com/architecture/choose-design-pattern-agentic-ai-system)

---

## 3. Tools, Function Calling & Interoperability

Sources describing *how agents connect to tools, APIs, and other agents*.

* **Function Calling — Gemini API**
  Structured tool invocation within LLM calls.
  [https://ai.google.dev/gemini-api/docs/function-calling](https://ai.google.dev/gemini-api/docs/function-calling)

* **Model Context Protocol (MCP)**
  Open standard for tool discovery and invocation.
  [https://github.com/modelcontextprotocol/](https://github.com/modelcontextprotocol/)

* **Agents Are Not Tools (A2A Conceptual Framing)**
  Conceptual distinction between agent–tool and agent–agent interaction.
  [https://discuss.google.dev/t/agents-are-not-tools/192812](https://discuss.google.dev/t/agents-are-not-tools/192812)

> Note: No stable, final public specification for an “Agent2Agent protocol” currently exists. References are conceptual rather than normative.

---

## 4. Evaluation, Quality & Agent Ops

Sources addressing *measurement, evaluation, and debugging of agentic systems*.

* **GenAI in Production: MLOps or GenAIOps?**
  Overview of operational evolution toward agent-centric systems.
  [https://medium.com/@sokratis.kartakis/genai-in-production-mlops-or-genaiops-25691c9becd0](https://medium.com/@sokratis.kartakis/genai-in-production-mlops-or-genaiops-25691c9becd0)

* **OpenTelemetry and AI Agent Observability**
  High-fidelity tracing for agent execution paths.
  [https://opentelemetry.io/blog/2025/ai-agent-observability/](https://opentelemetry.io/blog/2025/ai-agent-observability/)

---

## 5. Deployment & Production Infrastructure

Sources covering *runtime, scaling, and deployment options*.

* **Vertex AI Agent Engine Overview**
  Managed runtime for deploying and operating agents.
  [https://cloud.google.com/agent-builder/agent-engine/overview](https://cloud.google.com/agent-builder/agent-engine/overview)

* **Cloud Run vs GKE Concepts**
  Container-based deployment foundations applicable to agent services.
  [https://cloud.google.com/kubernetes-engine/docs/concepts/gke-and-cloud-run](https://cloud.google.com/kubernetes-engine/docs/concepts/gke-and-cloud-run)

* **Agent Starter Pack (Google Cloud)**
  Reference implementations and CI/CD examples.
  [https://github.com/GoogleCloudPlatform/agent-starter-pack](https://github.com/GoogleCloudPlatform/agent-starter-pack)

---

## 6. Security, Identity & Governance

Verified sources on *security, identity, and risk mitigation*.

* **Focus on Agents — Secure AI Framework (SAIF)**
  Threat models and defense-in-depth strategies for agents.
  [https://saif.google/focus-on-agents](https://saif.google/focus-on-agents)

* **Prompt Injection Attacks Against LLMs**
  Canonical reference on prompt-based security risks.
  [https://simonwillison.net/series/prompt-injection/](https://simonwillison.net/series/prompt-injection/)

* **SPIFFE: Secure Production Identity Framework**
  Cryptographic identity for non-human principals, including agents.
  [https://spiffe.io/](https://spiffe.io/)

* **Model Armor (Google Cloud)**
  Managed service for prompt and response security.
  [https://cloud.google.com/security-command-center/docs/model-armor-overview](https://cloud.google.com/security-command-center/docs/model-armor-overview)


---

