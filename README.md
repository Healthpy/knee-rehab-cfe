# 🤖 Rehab-CfE: An Adaptive Multi-Objective Framework for Group-Based Counterfactual Explanations in Rehabilitation

This repository implements an **adaptive multi-objective framework** for generating **group-based counterfactual explanations** in multivariate time series data, specifically designed for sensor-guided rehabilitation applications. Our approach addresses the critical need for clinically interpretable AI explanations in healthcare by integrating domain-informed sensor grouping with dynamic optimization techniques.

## 🎯 Overview

Traditional counterfactual explanation methods for time series data operate at individual feature levels, limiting their interpretability in clinical settings where experts reason about anatomical regions and sensor groups. Our framework introduces:

- **Domain-informed grouping** of IMU sensors into clinically meaningful units
- **Learnable group gates** for dynamic relevance determination
- **Adaptive multi-objective optimization** balancing validity, sparsity, and plausibility
- **Shapley-based feature ranking** for structured counterfactual generation

## 📊 Key Contributions

### 🔬 Technical Innovations
- **Adaptive Multi-Objective Framework**: Integrates Shapley-based group ranking with dynamic sensor selection
- **Learnable Group Gates**: Automatic relevance determination for sensor groups during optimization
- **Structured Feature Selection**: Domain-aligned grouping that reflects clinical understanding
- **Dynamic Weight Management**: Adaptive loss weighting that evolves during optimization

### 📈 Performance Achievements
- **67.6% channel sparsity** vs 0.0% for baseline M-CELS
- **86% feature sparsity** vs 72% for M-CELS
- Maintained comparable validity while significantly improving interpretability
- Successfully identifies clinically relevant movement patterns

## 🏥 Clinical Applications

### Movement Types Analyzed
- **Squat**: Weight-shift and forward-lean error patterns
- **Knee Extension**: Range-of-motion and lateral deviation errors  
- **Gait**: Stance-phase and trajectory errors

### Sensor Configuration
- **8 IMU sensors** placed bilaterally on key muscle groups:
  - Rectus Femoris (RF)
  - Hamstring (HAM) 
  - Tibialis Anterior (TA)
  - Gastrocnemius (GAS)
- **48 total channels** (8 sensors × 6 channels each: 3 accelerometer + 3 gyroscope)
- **Grouping strategies**: Sensor-level (8 groups) and modality-level (16 groups)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.8+

### Setup
```bash
# Clone the repository
git clone https://github.com/Healthpy/knee-rehab-cfe.git
cd knee-rehab-cfe

# Create conda environment
conda create -n rehab-cfe python=3.8
conda activate rehab-cfe

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Training Models
```bash
# Train movement-specific FCN models
python train_fcn.py --movement squat --epochs 50
python train_fcn.py --movement extension --epochs 50  
python train_fcn.py --movement gait --epochs 50
```

### Running Counterfactual Analysis
```bash
# Run comprehensive experiments
python scripts/run_comprehensive_experiments.py

# Run XAI analysis for specific movement
python scripts/run_analysis.py --movement-type squat --log-level INFO
```

### Visualization
```bash
# Visualize counterfactual results
python visualize_counterfactuals.py --movement squat --method adaptive-mo

# Compare different movements
python compare_movements.py
```

## 📋 Core Methodology

### 1. Shapley-Based Group Importance Ranking
Our framework computes channel-level Shapley values and aggregates them to group level:

```python
φ_g = (1/|g|) * Σ(φ_i for i in g)
```

where `φ_g` represents group importance and `|g|` is the group size.

### 2. Multi-Objective Optimization
The framework optimizes five complementary loss components:

```python
L_total = w₁*L_target + w₂*L_sparsity + w₃*L_smooth + w₄*L_group + w₅*L_gates
```

- **L_target**: Target class achievement
- **L_sparsity**: Feature-level sparsity promotion  
- **L_smooth**: Temporal coherence
- **L_group**: Group-level structured selection
- **L_gates**: Learnable gate regularization

### 3. Dynamic Group Gating
Learnable gates `θ_g` control sensor group relevance:

```python
M̃ = M ⊙ (Σ σ(θ_g) * G_g)
```

Groups with `σ(θ_g) < 0.5` are considered for removal during optimization.

## 📊 Dataset: KneE-PAD

### Data Characteristics
- **31 subjects** with left/right knee injuries
- **3 rehabilitation exercises** with correct and incorrect variations
- **IMU sampling**: 148.148 Hz (8 sensors × 6 channels = 48 total)
- **sEMG sampling**: 1,259.259 Hz (8 channels)

### Exercise Protocols
| Exercise | Correct Execution | Error Variations |
|----------|------------------|------------------|
| **Squat** | Descend to chair, return to standing | Weight-shift (WT), Forward-lean (FL) |
| **Extension** | Seated leg extension | No-full extension (NF), Lateral-lean (LL) |  
| **Gait** | 3m walk, turn, return | No-full extension (NF), Hip-abduction (HA) |

## 📈 Performance Results

### Comparison with M-CELS Baseline

| Metric | M-CELS | Adaptive-MO | Improvement |
|--------|--------|-------------|-------------|
| **Channel Sparsity** | 0.0% | **67.6%** | +67.6% |
| **Feature Sparsity** | 72% | **86%** | +14% |
| **Validity** | 93.3% | 90.2% | -3.1% |
| **L₂ Distance** | 13.4 | **11.0** | -18% |

### Movement-Specific Results

| Movement | Validity | Feature Sparsity | Channel Sparsity | Generation Time |
|----------|----------|------------------|------------------|-----------------|
| **Squat** | 98.7% | 88% | 72.9% | 5.79s |
| **Extension** | 93.3% | 87% | 68.6% | 9.37s |
| **Gait** | 56.0% | 79% | 56.6% | 7.74s |

## 🏗️ Project Structure

```
├── src/
│   ├── core/           # Base classes and utilities
│   ├── models/         # FCN model definitions  
│   ├── explainers/     # Counterfactual algorithms
│   ├── data/          # Data processing utilities
│   ├── experiments/   # Experiment management
│   └── evaluation/    # Metrics and analysis
├── scripts/           # Execution scripts
├── models/           # Trained model files
├── results/          # Experimental results
├── docs/            # Documentation and paper
└── data/           # IMU and EMG datasets
```

## 🔬 Key Classes and Methods

### Core Framework
- `AdaptiveMultiObjectiveExplainer`: Main counterfactual generation class
- `ShapleyGroupRanker`: Computes group-level importance scores
- `DynamicGroupGates`: Learnable relevance determination
- `MultiObjectiveLoss`: Adaptive loss weighting system

### Evaluation Metrics
- `ValidityMetrics`: Target class achievement assessment
- `SparsityMetrics`: Feature and channel-level sparsity
- `PlausibilityMetrics`: LOF-based realism evaluation
- `RobustnessMetrics`: Noise stability analysis

## 📚 Dependencies

### Core Requirements
```
torch>=1.8.0
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
tsinterpret>=0.1.0
tslearn>=0.5.0
shap>=0.41.0
```
