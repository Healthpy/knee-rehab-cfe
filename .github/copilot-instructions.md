# Copilot Instructions for XAI Counterfactual Analysis

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Project Overview
This is an XAI (Explainable AI) counterfactual analysis project focused on IMU/EMG data for knee rehabilitation. The project implements various explainability methods to understand how machine learning models make predictions about movement patterns and injury recovery.

## Code Guidelines
- Follow PEP 8 Python style guidelines
- Use type hints for all function parameters and return values
- Include comprehensive docstrings for all classes and methods
- Implement logging instead of print statements for debugging
- Use configuration files instead of hardcoded values
- Follow the Single Responsibility Principle for all classes and functions

## Project Structure
- `src/core/`: Base classes and utilities
- `src/models/`: Machine learning model definitions and utilities
- `src/explainers/`: Counterfactual explanation algorithms
- `src/data/`: Data loading, preprocessing, and synthetic data generation
- `src/analysis/`: Analysis tools and visualization components
- `src/experiments/`: Experiment management and execution
- `config/`: Configuration files for models, data, and experiments
- `tests/`: Unit and integration tests

## Technical Context
- IMU data: 8 sensors, 6 channels each (3 accelerometer + 3 gyroscope), 48 total channels
- EMG data: Surface electromyography signals for muscle activity
- Movement types: squat, extension, gait
- Injury classification: Left vs Right knee injuries
- Models: Fully Connected Networks (FCN) for time series classification

## Key Concepts
- Counterfactual explanations: "What would need to change for a different prediction?"
- Saliency methods: Understanding which features are most important
- Adaptive explanations: Explanations that consider injury side and movement type
- Sensor importance: Identifying which IMU sensors contribute most to predictions

## Best Practices
- Always validate data shapes and types before processing
- Include error handling for file I/O operations
- Use seed values for reproducible experiments
- Cache expensive computations when possible
- Create modular, reusable components
- Test with both synthetic and real data
