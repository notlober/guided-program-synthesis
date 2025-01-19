# Guided Program Synthesis for MNIST Classification

This project demonstrates a simple guided program synthesis approach applied to the MNIST dataset. A model (Oracle) is trained to synthesize a binary program that classifies handwritten digits. The synthesized program is saved and used for inference.

## Features
- **Oracle Model**: A neural network that generates binary program rules.
- **ProgramState**: Applies the learned rules to classify MNIST digits.
- **Training**: Train the Oracle to find optimal program rules.
- **Inference**: Use the saved program to classify MNIST test data.

## Files
- `main.py`: Training script for generating the binary program.
- `inference.py`: Inference script to test the binary program on MNIST test data.

## Usage

1. **Train the Oracle**:
   ```bash
   python main.py
   ```
   The trained binary program will be saved as `binary_program.pth`.

2. **Run Inference**:
   ```bash
   python inference.py
   ```
   The script will load the binary program and calculate test accuracy.

## Output
- **Training**: Displays loss, training accuracy, and active rules at each step.
- **Inference**: Outputs the final test accuracy.

## Explanation
This project trains an Oracle model to generate binary program rules for digit classification. The ProgramState module applies these rules to one-hot-encoded MNIST data. The process combines machine learning and programmatic reasoning.