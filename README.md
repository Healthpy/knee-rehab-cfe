# Adaptive Group-Based Counterfactual Explanations for Time-Series Rehabilitation Data

This repository implements an **adaptive group-based counterfactual explanation framework** for multivariate time series, specifically designed for IMU-based rehabilitation movement analysis. The framework addresses the challenge of generating clinically interpretable explanations by aligning counterfactuals with anatomical sensor groups (muscle-level IMU units) rather than individual channels, producing sparse, biomechanically coherent guidance for rehabilitation exercises.

## 🎯 Overview

Traditional counterfactual explanation methods for time series data operate at individual feature levels, limiting their interpretability in clinical settings where experts reason about anatomical regions and sensor groups. Our framework introduces:

- **Domain-informed grouping** of IMU sensors into clinically meaningful units
- **Learnable group gates** for dynamic relevance determination
- **Adaptive multi-objective optimization** balancing validity, sparsity, and plausibility
- **Shapley-based feature ranking** for structured counterfactual generation

## Key Contributions

1. **Adaptive Multi-Objective Framework**: A two-stage approach combining Shapley-based group ranking with learnable gate mechanisms for structured counterfactual generation in high-dimensional IMU data

2. **Learnable Gate (LG) Mechanism**: Trainable per-group relevance gates that are jointly optimized with perturbation masks, enabling automatic selection of sparse and clinically meaningful sensor groups

3. **Shapley-Adaptive (SA) Ablation**: Demonstrates that Shapley-based ranking alone maintains validity (77-93% across ratios 0.3-0.9) but fails to enforce group sparsity, motivating explicit learnable group selection

4. **Comprehensive Evaluation on KneE-PAD**: On 8 IMU sensors (48 channels), 31 participants, 9 exercise error classes, LG (SHAP pruned) achieves:
   - **27% better group sparsity** than M-CELS baseline (8.2 vs 11.2 modality groups)
   - **94.7% validity** vs 90.0% for M-CELS
   - **Faster generation time** (9.7s vs 10.0s)
   - **Preserved temporal smoothness**

5. **Exercise-Specific Analysis**: Group-structured counterfactuals yield muscle-specific clinical guidance for squat, knee extension, and gait rehabilitation tasks

## Clinical Applications

### Movement Types Analyzed
- **Squat**: Weight-shift (SquatWT) and forward-lean (SquatFL) error patterns
- **Knee Extension**: No-full extension (ExtNF) and lateral-lean (ExtLL) errors  
- **Gait**: No-full extension (GaitNF) and hip-abduction (GaitHA) errors

### Sensor Configuration
- **8 IMU sensors** placed bilaterally on key muscle groups:
  - Rectus Femoris (RF)
  - Hamstring (HAM) 
  - Tibialis Anterior (TA)
  - Gastrocnemius (GAS)
- **48 total channels** (8 sensors × 6 channels each: 3 accelerometer + 3 gyroscope)
- **16 modality groups** (accelerometer and gyroscope separated per muscle)
- **Sampling rate**: 148.148 Hz

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)

### Setup
```bash
# Create conda environment
conda create -n knee-rehab python=3.8
conda activate knee-rehab

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Training Models
```bash
# Train FCN model with subject-disjoint split
python src/src/models/train_fcn_subject_split.py

# Train with trial-based split
python src/models/train_fcn_trial_split.py

# Train TCN model
python src/models/train_tcn.py
```

### Running Counterfactual Analysis
```bash
# Evaluate Learnable Gate method (subject split)
python src/models/evaluate_fcn_imu_learnable_gate_subject_split.py

# Evaluate Shapley-Adaptive method
python src/models/evaluate_fcn_imu_sa_subject_split.py

# Evaluate M-CELS baseline
python src/models/evaluate_fcn_imu_mcels_subject_split.py

# Run ablation study
python src/models/ablation_study_paper_grade.py

# Analyze exercise-specific patterns
python src/models/analyze_exercise_specific_cfe.py
```

## Core Methodology

### 1. Shapley-Based Group Importance Ranking
Channel-level Shapley values are computed using GradientSHAP and aggregated to group level using maximum absolute value:

```python
Φ_g = max_{i ∈ g} |φ_i|
```

The top-k groups are selected for optimization (k = ⌊r·K⌋, where r=0.8).

### 2. Multi-Objective Optimization
The framework optimizes four complementary loss components:

```python
L_total = w₁·L_target + w₂·L_sparsity + w₃·L_smooth + w₄·L_gates
```

- **L_target**: Target class achievement (1 - p_target)
- **L_sparsity**: Feature-level sparsity (L1 norm on mask)
- **L_smooth**: Temporal coherence (total variation)
- **L_gates**: Group gate regularization (L1 + binarization)

Weights are adapted dynamically based on target probability.

### 3. Learnable Group Gates
Trainable gates θ_g control sensor group relevance with sigmoid activation. Gates are regularized to encourage sparsity and binary values (0 or 1). Post-optimization pruning removes groups with low gate values (< 0.3 threshold).

### Example: Counterfactual Explanation in Action

The following example demonstrates how our approach generates counterfactual explanations for gait correction, transforming "Walking - No full extension" to "Walking - Correct" by modifying specific sensor channels:

![Counterfactual Example: Gait Correction](results\evaluation\learnable_gate\visualizations\example_1_class7_to_6.png)

*Figure 1: Time-series visualization showing counterfactual transformation from incorrect gait (no full extension) to correct walking pattern. The plot shows accelerometer (left) and gyroscope (right) data across 8 IMU sensors, with original data in blue, counterfactual in green, and perturbation mask highlighting modified regions.*

## 📊 Dataset: KneE-PAD

### Data Characteristics
- **31 subjects** with knee pathologies (left/right knee injuries)
- **3 rehabilitation exercises** with 3 execution types each (1 correct + 2 error variants)
- **9 total classes** for classification
- **IMU sampling**: 148.148 Hz (8 sensors × 6 channels = 48 channels)
- **Subject-disjoint splits**: 70/15/15 train/val/test ratio

### Exercise Protocols
| Exercise | Correct Execution | Error Variations |
|----------|------------------|------------------|
| **Squat** | Descend to chair, return to standing | Weight-shift (SquatWT), Forward-lean (SquatFL) |
| **Extension** | Seated leg extension | No-full extension (ExtNF), Lateral-lean (ExtLL) |  
| **Gait** | 3m walk, turn, return | No-full extension (GaitNF), Hip-abduction (GaitHA) |

### Sensor Placement
8 IMU sensors placed bilaterally on:
- Rectus Femoris (RF) - right/left
- Hamstrings (HAM) - right/left
- Tibialis Anterior (TA) - right/left
- Gastrocnemius (GAS) - right/left

Each sensor provides 3-axis accelerometer + 3-axis gyroscope = 6 channels per sensor.

## 📈 Performance Results

### Comparison with M-CELS Baseline (n=150 test samples, subject-disjoint split)

| Metric | M-CELS | LG (SHAP pruned) | Improvement |
|--------|--------|------------------|-------------|
| **Validity** | 90.0% | **94.7%** | +4.7% |
| **Modality Group Sparsity** | 11.2 groups | **8.2 groups** | **-27%** |
| **Channel Sparsity** | 23.2 channels | **18.0 channels** | -22% |
| **Generation Time** | 10.0s | **9.7s** | -3% |
| **Target Confidence** | **88.2%** | 80.8% | -7.4% |

### Exercise-Specific Results

| Exercise | Method | Validity | Modality Groups | Gen Time |
|----------|--------|----------|-----------------|----------|
| **Squat** | LG | 100% | 8.8 | 8.5s |
|  | M-CELS | 100% | 13.3 | 9.0s |
| **Extension** | LG | **86.3%** | 6.7 | 8.8s |
|  | M-CELS | 79.5% | 9.7 | 8.9s |
| **Gait** | LG | 100% | 10.1 | **7.0s** |
|  | M-CELS | 100% | 15.2 | 10.9s |

**Key finding**: LG achieves ~33% better group sparsity across all exercises, with particularly strong improvements on knee extension (+6.8% validity).

### Visual Comparison: Modality Group Activation

The following heatmap compares the modality group activation patterns between our Learnable Gate (LG) method and the M-CELS baseline across different exercise types:

![Modality Group Activation Comparison](results\experiments\exercise_specific\exercise_modality_activation_heatmap.png)

*Figure 2: Comparison of modality group activation frequencies between LG (SHAP pruned) and M-CELS methods across exercise types (Squat, Extension, Gait). The heatmap shows activation percentages for each sensor modality group (RF, HAM, TA, GAS with accelerometer/gyroscope splits for left/right sides), demonstrating LG's superior group sparsity with more selective sensor activation.*

## 🏗️ Project Structure

```
├── src/src/
│   ├── architectures/    # Neural network models (FCN, TCN)
│   ├── models/           # Training and evaluation scripts
│   ├── explainer/        # Counterfactual generation methods
│   ├── data/            # Data processing and loading
│   ├── utils/           # Helper functions
│   └── visualization/   # Plotting and view generation
├── models/              # Pre-trained model checkpoints
├── results/             # Experimental results and metrics
│   ├── ablation/        # Ablation study results
│   ├── evaluation/      # Method comparison results
│   └── experiments/     # Exercise-specific analysis
├── docs/                # Documentation and paper
└── config/              # Configuration files
```

## 🔬 Key Methods

### Counterfactual Explainers
- **Learnable Gate (LG)**: Trainable group gates with SHAP initialization
- **Shapley-Adaptive (SA)**: Static SHAP-based group ranking
- **M-CELS Baseline**: Channel-level counterfactual generation

## 📚 Dependencies

Core libraries:
- PyTorch >= 1.9.0
- NumPy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- SHAP >= 0.41.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

See [requirements.txt](requirements.txt) for complete list.

<!-- ## 🤝 Citation

If you use this work, please cite: -->
<!-- 
```bibtex
@article{chukwu2026adaptive,
  title={Adaptive Group-Based Counterfactual Explanations for Time-Series Rehabilitation Data},
  author={Chukwu, Emmanuel C. and Schouten, Rianne M. and Tabak, Monique and Pechenizkiy, Mykola},
  journal={IEEE Conference Proceedings},
  year={2026}
}
``` -->

## 🔗 Related Work

### Counterfactual Methods
- **M-CELS** [Li et al., 2024]: Multivariate time-series counterfactuals
- **CoMTE** [Ates et al., 2021]: Instance-based counterfactuals
- **Native Guide** [Delaney et al., 2021]: Nearest unlike neighbor

### Rehabilitation & IMU Analysis  
- **KneE-PAD Dataset** [Kasnesis et al., 2025]: Knee rehabilitation IMU data
- **IMU Calibration** [Bonfiglio et al., 2024]: Sensor-to-segment alignment
- **Clinical IMU Analysis** [Routhier et al., 2020; Porciuncula et al., 2018]

## 📄 License

This project is licensed under the MIT License.

<!-- ## 🙏 Acknowledgments

- **KneE-PAD Dataset**: Kasnesis et al., 2025
- **Eindhoven University of Technology**: Department of Mathematics and Computer Science
- **University of Twente**: Biomedical Signals and Systems Group -->

<!-- ## Contact

**Emmanuel C. Chukwu** (e.c.chukwu@tue.nl)  
Department of Mathematics and Computer Science
Eindhoven University of Technology -->