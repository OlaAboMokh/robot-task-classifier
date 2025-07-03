# robot-task-classifier
A full machine-learning pipeline that classifies robot episodes as **BoxCleanup** or **DrawerCleanup** using only internal sensor and action data. The workflow covers data aggregation, PCA, multiple classifiers, feature-importance analysis, and evaluation with no data leakage.

---

## What the project does
We analyze sensor/control data from the GR00T-X Embodiment Sim dataset and train several models to predict the task type (box vs. drawer) purely from internal features.

Key explorations:

- Performance of five classifiers (LogReg, Lasso, Linear SVM, Random Forest, Gradient Boosting) before and after PCA
- Feature differences between tasks  
- Dimensionality reduction with PCA  
- Feature importance for each model  

---

## Models used
Each model is trained **before and after PCA** (95 % variance retained):

- Logistic Regression  
- Lasso Logistic Regression (feature selection)  
- Linear SVM  
- Random Forest  
- Gradient Boosting  

---

## Evaluation methods
- ROC-AUC (cross-validated & test)  
- Confusion Matrices  
- Log-loss curves  
- Classification reports (precision / recall / F1)  
- Feature-importance plots (coefficients & Gini)  
- PCA component analysis (PC1 loadings, state vs. action)  

---

## Folder structure
```text
robot-task-classifier/
├── src/
│   └── main.py            # Full ML pipeline
├── data/
│   ├── BoxCleanup/        # .parquet files
│   └── DrawerCleanup/     # .parquet files
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore

```

## How to Run

1. **Clone or download** this repository.  
2. **Place your GR00T dataset** (parquet files) inside the `data/` folder, under the appropriate subfolders:
   - `data/BoxCleanup/`
   - `data/DrawerCleanup/`
3. **Install the required packages** and run the pipeline:

```bash
pip install -r requirements.txt
python src/main.py

```

## Requirements

| Package          | Tested Version |
| ---------------- | -------------- |
| Python           | 3.8 – 3.12     |
| pandas           | 2.2            |
| numpy            | 1.26           |
| scipy            | 1.12           |
| pyarrow          | 15.0           |
| scikit-learn     | 1.5            |
| matplotlib       | 3.9            |
| seaborn          | 0.13           |
| statsmodels      | 0.15           |
| glob2 (optional) | 0.7            |
| tqdm  (optional) | 4.66           |

Install everything with:
```bash
pip install -r requirements.txt
```
## License
MIT – feel free to use, modify, and share.

## Credits
Built by Ola Abo Mokh using the GR00T-X Embodiment Sim dataset.
Thanks to the open-source Python ML ecosystem.

```
