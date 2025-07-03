# robot-task-classifier
A full machine learning pipeline that classifies robot episodes as BoxCleanup or DrawerCleanup using only internal sensor and action data. Includes data aggregation, PCA, multiple classifiers, feature importance, and clean evaluation with no data leakage.
---

## 🧠 What the project does

We analyze sensor and control data collected from a robot performing different tasks and train several models to predict the task type (box vs drawer) based purely on internal features.

We also explore:
- How features differ between tasks
- Dimensionality reduction using PCA
- Which features are most important for prediction

---

## 🧪 Models used

Each model is trained both **before and after PCA** (retain 95% variance):

- Logistic Regression  
- Lasso Logistic Regression (with feature selection)  
- Linear SVM  
- Random Forest  
- Gradient Boosting  

---

## 📊 Evaluation methods

- ROC-AUC scores (cross-validated and test)
- Confusion Matrices  
- Log-loss curves  
- Classification reports (precision, recall, F1)  
- Feature importance (coefficients, Gini)  
- PCA component analysis (PC1 loadings, state vs action)  

---

## 🗂 Folder Structure
robot-task-classifier/
├── src/
│ └── main.py # Full ML pipeline
├── data/ # Place parquet files here
├── figures/ # Auto-generated plots (ignored by Git)
├── README.md # This file
├── requirements.txt # Python dependencies
└── .gitignore # Files/folders Git should ignore

---

## ▶️ How to run  

1. Clone or download this repo.  
2. Place your GR00T dataset (parquet files) in the `data/` folder.  
3. Run the following commands:

```bash
pip install -r requirements.txt
python src/main.py
```

## 📦 Requirements

| Package        | Tested Version |
|----------------|----------------|
| Python         | 3.8 – 3.12     |
| pandas         | 2.x            |
| numpy          | 1.26           |
| scikit-learn   | 1.5            |
| matplotlib     | 3.9            |
| seaborn        | 0.13           |
| statsmodels    | 0.15           |

Install everything with:
```bash
pip install -r requirements.txt
```
## 📄 License
MIT – feel free to use, modify, and share.

## Credits
Built by Ola Abo Mokh using the GR00T-X Embodiment Sim dataset.
Thanks to the open-source Python ML ecosystem.

```
Let me know if you’d like this saved as a file for download, or if you want help uploading it to your GitHub repo now.
