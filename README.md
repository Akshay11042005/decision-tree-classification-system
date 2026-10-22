# Decision Tree Classification System

An interactive Streamlit application for CSV-based classification using two educational decision-tree learners:
- **ID3-style:** selects splits by information gain.
- **C4.5-style:** selects splits by gain ratio.

## Features
- CSV upload and dataset preview
- Target-column selection
- Missing-value handling and categorical encoding
- Train/test split
- Decision-tree visualization
- Predictions for user-entered examples
- Confusion matrices and classification reports
- Accuracy, weighted precision, weighted recall, and weighted F1 comparison
- Downloadable metrics CSV

## Requirements
- Python 3.10 or newer recommended
- pip
- Graphviz package is listed for extensibility; current tree rendering is done with Matplotlib.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## GitHub setup

Create an empty GitHub repository named `decision-tree-classification-system`, then run:

```bash
git init
git add .
git commit -m "Initial decision tree classification system"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/decision-tree-classification-system.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.

## How to use
1. Launch the Streamlit app.
2. Upload a CSV dataset, or click **Load sample dataset**.
3. Select the target/class column.
4. Set test size and tree settings in the sidebar.
5. Open **Train & evaluate** and click **Train ID3 and C4.5**.
6. Inspect the trees, confusion matrix, classification report, and metric comparison.
7. Open **Predict** to enter a new example and compare predictions.
8. Download the comparison metrics as CSV.

## Algorithm notes and limitations
This project is an educational implementation of information-gain and gain-ratio decision trees. It searches numeric thresholds in the preprocessed feature matrix. Categorical features are one-hot encoded before tree training. Therefore, the C4.5-style model is **not a complete reproduction of Quinlan's original C4.5**, which also includes native continuous-attribute handling, missing-value routing, and pruning. The same preprocessing and train/test split are used for both models to make their comparison more consistent.

## Project structure

```text
decision-tree-classification-system/
├── app.py
├── algorithms.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Suggested future improvements
- Implement native categorical splits and C4.5 pruning
- Add cross-validation and hyperparameter tuning
- Export the tree as Graphviz DOT/PDF
- Add dataset download and saved experiment history
- Add unit tests for entropy, information gain, gain ratio, and prediction paths
