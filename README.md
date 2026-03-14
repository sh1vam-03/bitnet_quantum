# ⚛️ Quantum-Assisted BitNet: QACWS Experiment

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19024106.svg)](https://doi.org/10.5281/zenodo.19024106)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![IBM Quantum](https://img.shields.io/badge/IBM%20Quantum-ibm__torino-6929c4.svg)](https://quantum.ibm.com)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**A Preliminary Investigation of Quantum-Assisted Ternary Weight Optimization for BitNet Neural Networks**

> Proof-of-concept experiments running a hybrid quantum-classical pipeline on real IBM Quantum hardware (IBM Torino, 133 qubits).

📄 **Paper:** [https://doi.org/10.5281/zenodo.19024106](https://doi.org/10.5281/zenodo.19024106)  
👤 **Author:** Sh1vam — [sh1vam-03](https://github.com/sh1vam-03)  
📅 **Published:** March 2026

---

## 🧠 What Is This?

[BitNet](https://arxiv.org/abs/2310.11453) neural networks use ternary weights `{-1, 0, +1}` instead of full floating-point numbers — achieving **20x memory compression**. The problem is that classical gradient descent **gets stuck** when training these models because the `sign()` function has near-zero gradients everywhere.

This project proposes **QACWS (Quantum-Assisted Critical Weight Search)** — a hybrid pipeline that uses a real quantum computer to escape the saddle points where classical training gets trapped.

```
Classical training gets stuck at 56.8% accuracy
           ↓
Gradient analysis finds the 16 hardest weights
           ↓
Quantum circuit searches those 16 weights on IBM Torino
           ↓
Hybrid pipeline achieves 59.0% accuracy ✅
```

---

## 🏆 Key Results

| Method | Accuracy | Loss | Hardware |
|--------|----------|------|----------|
| Classical gradient descent | 56.8% | 0.6931 | CPU |
| Classical random search | 58.4% | 1.1324 | CPU |
| **QACWS Hybrid (ours)** | **59.0%** | **0.6993** | **IBM Torino QPU** |

> **+2.2% above the classical ceiling** that gradient descent could not break through even after 800 epochs.

### IBM Quantum Job IDs (fully reproducible)
| Round | Job ID | Unique States | QPU Time |
|-------|--------|---------------|----------|
| Round 1 | `d6qpj5i0q0ls73cskod0` | 181 | 13.7s |
| Round 2 | `d6qpj87r88ds73dcms9g` | 3,860 | 10.4s |

---

## 🔬 The 5-Version Journey

Each version exposed a different failure mode — documented transparently:

| Version | Strategy | Result | What We Learned |
|---------|----------|--------|-----------------|
| v1 | Raw circuit | Runtime error | Must transpile to native gates |
| v2 | 2 qubits/weight | 51.0% / 96% zeros | Dual encoding too sparse |
| v3 | 1 qubit/weight | 49.6% / loss 9.13 | Phase bias catastrophic |
| v4 | Centered phases | 50.4% / loss 0.69 | Hardware readout bias |
| **v5** | **QACWS hybrid** | **59.0% ✅** | **Critical targeting works** |

---

## 🛠️ How QACWS Works

```
Step 1: Classical pre-training (600 epochs)
        → Model plateaus at 56.8%, stuck in saddle point

Step 2: Gradient magnitude analysis
        → Find 16 weights with highest gradient
        → These are 15.5x harder than average for gradient descent

Step 3: Quantum circuit on IBM Torino (133 qubits)
        → 1 qubit = 1 real weight (no fake expansion!)
        → QAOA-inspired: H gates → RZ phase → CX entangle → H interference
        → 2 rounds, 4,096 shots each
        → Round 2 explores 3,860 unique configurations

Step 4: Classical fine-tuning from quantum solution
        → Polishes the quantum-found configuration
        → Final result: 59.0% accuracy
```

---

## 📁 Repository Structure

```
bitnet_quantum/
├── bitnet_quantum_v2.py      # v2: 2 qubits per weight
├── bitnet_quantum_v3.py      # v3: 1 qubit, biased phases
├── bitnet_quantum_v4.py      # v4: centered phases + threshold
├── bitnet_quantum_v5.py      # v5: QACWS true hybrid ← main experiment
├── quantum_results_v2.json   # v2 IBM QPU results
├── quantum_results_v3.json   # v3 IBM QPU results
├── quantum_results_v4.json   # v4 IBM QPU results
├── quantum_results_v5.json   # v5 IBM QPU results ← final results
├── bitnet_quantum.pdf        # Research paper
├── requirements.txt          # Python dependencies
├── .env.example              # API key template
└── test.py                   # IBM Quantum connection test
```

---

## 🚀 Quickstart

### 1. Clone and setup
```bash
git clone https://github.com/sh1vam-03/bitnet_quantum
cd bitnet_quantum
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure IBM Quantum credentials
```bash
cp .env.example .env
# Edit .env and add your IBM Quantum API key and CRN
```

Or get free access at [quantum.ibm.com](https://quantum.ibm.com)

### 3. Test your connection
```bash
python test.py
# Should print: ✅ Connected! 3 backends available
```

### 4. Run the main experiment (v5)
```bash
python bitnet_quantum_v5.py
# Takes ~25-35 minutes (classical + 2 QPU rounds + finetune)
# Results saved to quantum_results_v5.json
```

---

## 📦 Requirements

```
qiskit
qiskit-ibm-runtime
numpy
```

```bash
pip install -r requirements.txt
```

**Hardware:** IBM Quantum account with access to `ibm_torino` or any 100+ qubit backend.

---

## ⚠️ Limitations

This is a **preliminary proof-of-concept**. Before stronger claims can be made:

- [ ] Statistical significance testing across 10+ runs
- [ ] Simulated annealing baseline comparison
- [ ] Noiseless simulator vs real QPU comparison
- [ ] Scaling to real ML benchmarks (MNIST, text classification)
- [ ] Readout error mitigation

See Section 6 of the paper for full honest assessment.

---

## 📊 Raw Results

All experimental results are in the JSON files. Example from v5:

```json
{
  "experiment": "BitNet Quantum-Assisted Critical Weight Search",
  "version": "v5",
  "backend": "ibm_torino",
  "qubits_used": 16,
  "shots": 4096,
  "classical_only_baseline": {"acc": 0.568, "loss": 0.6931},
  "quantum_ibm_qpu": {"acc": 0.590, "loss": 0.6993},
  "step4_classical_finetune": {"acc": 0.590, "loss": 0.6993}
}
```

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@misc{sh1vam2026qacws,
  author    = {Sh1vam},
  title     = {A Preliminary Investigation of Quantum-Assisted 
               Ternary Weight Optimization for BitNet Neural Networks},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19024106},
  url       = {https://doi.org/10.5281/zenodo.19024106}
}
```

---

## 🔮 Future Work

- Run same experiment on **D-Wave hybrid solver** (2M+ variables)
- Add **simulated annealing baseline** for honest comparison
- Scale to **MNIST binary classification**
- Apply **readout error mitigation** to reduce hardware noise
- Extend QACWS to other discrete neural architectures

---

## 📬 Contact

**Sh1vam** — Independent Researcher, India  
GitHub: [@sh1vam-03](https://github.com/sh1vam-03)  
Paper: [https://doi.org/10.5281/zenodo.19024106](https://doi.org/10.5281/zenodo.19024106)

---

## 📜 License

This work is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

You are free to share and adapt this work with appropriate credit.