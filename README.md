# knee-rehab-cfe
# 🤖 Rehab-CfE: Counterfactual Explanations in Sensor-Guided Rehabilitation

This repository contains experiments and utilities for generating and visualizing **counterfactual explanations (CfEs)** using time-series sensor data collected from physical rehabilitation sessions. The goal is to support **interpretable feedback** for clinicians and patients by identifying and correcting faulty exercise execution.

We use the `tsinterpret` library to assess counterfactuals from multivariate time-series inputs, specifically from **48-channel IMU** data and **8-channel surface EMG (sEMG)** recordings per session.

---

## 📊 Project Goals

- Analyze rehabilitation exercise performance using high-dimensional sensor data
- Generate **counterfactual explanations** to show "what should have happened"
- Highlight affected channels (e.g., incorrectly used muscles, faulty movements)
- Provide intuitive **visualizations** for interpretability and clinical feedback

---

## 📚 Dependencies

- `tsinterpret`
- `tslearn`
- `numpy`, `pandas`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `torch` (if using deep CF models)
- `plotly` (for interactive visualizations)

Install using:

```bash
pip install -r requirements.txt