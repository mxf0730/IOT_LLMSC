# IoT-LLMSC: Cognitive Multi-Agent Simulation Framework for IoT-Informed Supply Chain Resilience
This repository hosts the official computational artifacts, system prompts, human validation corpus, and reproducible benchmarks for the IEEE Access manuscript:
> **"LLM-Enabled Multi-Agent Simulation Framework for IoT-Informed Supply Chain Resilience"** (Manuscript ID: Access-2026-39091).

---

## 📌 Architecture Overview

IoT-LLMSC addresses the cognitive gap in IoT-enabled supply chains by formalizing a decoupled **5-Tier Architecture (L0 to L4)**:
- **Layer 0 (Physical World)**: Discrete-event multi-echelon inventory conservation grounded in OR-Gym (`InvManagement-v1`) and Sterman's beer game dynamics.
- **Layer 1 (Perception Layer)**: Physical sensor degradation models including Gilbert-Elliott RFID packet-loss, Poisson GPS jitter, and threshold-activated thermal decay kinetics.
- **Layer 2 (Governance Middleware)**: Five distinct information accessibility regimes: M1 (Siloed), M2 (EDI), M3 (Delayed), M4 (Full Transparency), and M5 (Federated Knowledge).
- **Layer 3 (Cognitive Agents)**: Role-specific LLM agents equipped with a 3-tier hierarchical context engine (Working Memory, Episodic Buffer, Semantic Summarizer) and physical constraint alignment.
- **Layer 4 (Analytics & Control)**: Post-hoc LLM-as-Judge behavioral annotation (6+1 modes), causal mediation decomposition, and closed-loop Adaptive Trust-Weighted Coordination (ATWC).

---

## 🛠️ Quick Start

### 1. Environment Setup
```bash
git clone [https://github.com/your-anonymous-repo/IoT-LLMSC-Benchmark.git](https://github.com/https://github.com/mxf0730/IOT_LLMSC)
cd IoT-LLMSC
pip install -r requirements.txt
