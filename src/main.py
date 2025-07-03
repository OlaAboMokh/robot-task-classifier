# %% 📦 IMPORT LIBRARIES & SETUP
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import statsmodels.api as sm
import warnings
from sklearn.model_selection import StratifiedKFold
import random
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC 
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def plot_cv_curve(lr_cv, metric_label="Mean CV Accuracy", title=None):
    """
    Plots the average cross-validation score vs. C for a fitted
    LogisticRegressionCV object.

    Parameters
    ----------
    lr_cv : fitted sklearn.linear_model.LogisticRegressionCV
    metric_label : str
        Label for Y-axis (e.g., 'Mean CV Accuracy' or 'Mean ROC-AUC').
    title : str
        Plot title. If None, a default based on penalty is used.
    """
    all_scores = lr_cv.scores_  # dict of {class: [n_folds x n_Cs]}
    mean_scores = np.mean([scr.mean(axis=0) for scr in all_scores.values()], axis=0)
    Cs = lr_cv.Cs_

    plt.figure(figsize=(7, 4))
    plt.semilogx(Cs, mean_scores, marker="o", lw=2)
    plt.axvline(x=lr_cv.C_[0], color="red", linestyle="--", label=f"Selected C = {lr_cv.C_[0]:.4f}")
    plt.xlabel("Inverse Regularization Strength (C = 1/λ)")
    plt.ylabel(metric_label)
    if title is None:
        title = f"CV Curve for penalty='{lr_cv.penalty}'"
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

warnings.filterwarnings("ignore", category=FutureWarning)

# Set random seed for reproducibility
SEED = 20250424
def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

set_global_seeds(SEED)


# %% 📁 LOAD DATA FROM PARQUET FILES
# Define relative paths
box_path = os.path.join("data", "gr00t_dataset", "BoxCleanup")
drawer_path = os.path.join("data", "gr00t_dataset", "DrawerCleanup")

# Find all .parquet files
box_files = glob.glob(os.path.join(box_path, "**/*.parquet"), recursive=True)
drawer_files = glob.glob(os.path.join(drawer_path, "**/*.parquet"), recursive=True)

# Load and tag with task type
def load_parquets(files, task_type):
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["task_type"] = task_type
    return df

# Load both task types
box_df = load_parquets(box_files, "box")
drawer_df = load_parquets(drawer_files, "drawer")

# Combine into one DataFrame
full_df = pd.concat([box_df, drawer_df], ignore_index=True)

# ✅ Show results
print("✅ Loaded data:")
print(f"Box shape: {box_df.shape}")
print(f"Drawer shape: {drawer_df.shape}")
print(f"Full shape: {full_df.shape}\n")
print("Sample columns:", list(full_df.columns))

# %% 📊 Task label distribution
sns.countplot(x='task_type', data=full_df)
plt.title("Task Type Distribution")
plt.xlabel("Task")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# %% 📊 Count of episodes per task
episode_lengths = full_df.groupby(['task_type', 'episode_index']).size().reset_index(name='episode_length')
plt.figure(figsize=(5, 4))
sns.countplot(data=episode_lengths, x='task_type')
plt.title("Number of Episodes per Task")
plt.xlabel("Task Type")
plt.ylabel("Number of Episodes")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% 📏 EPISODE LENGTH ANALYSIS

# Basic stats
print("\n📊 Episode length statistics per task:")
print(episode_lengths.groupby("task_type")["episode_length"].describe())

# Optional: visualize with seaborn

plt.figure(figsize=(10,5))
sns.histplot(data=episode_lengths, x="episode_length", hue="task_type", kde=True, bins=50)
plt.title("Episode Length Distribution by Task Type")
plt.xlabel("Episode Length (Steps)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# %% T-test: Compare episode lengths between tasks

# Split episode lengths by task
box_lengths = episode_lengths[episode_lengths["task_type"] == "box"]["episode_length"]
drawer_lengths = episode_lengths[episode_lengths["task_type"] == "drawer"]["episode_length"]

# Run t-test
t_stat, p_value = ttest_ind(box_lengths, drawer_lengths)

print("\n🎯 T-Test Results:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("✅ The difference is statistically significant.")
else:
    print("❌ No significant difference.")

# 📊 Boxplot of episode lengths by task
plt.figure(figsize=(8, 5))
sns.boxplot(data=episode_lengths, x="task_type", y="episode_length", palette="Set2")
plt.title("Episode Length Distribution by Task")
plt.xlabel("Task")
plt.ylabel("Episode Length")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% Linear regression: episode_length ~ task_type

# Convert task_type to numeric: box=0, drawer=1
episode_lengths["task_type_num"] = episode_lengths["task_type"].map({"box": 0, "drawer": 1})

# Define regression model
X_linreg = sm.add_constant(episode_lengths["task_type_num"])
y_linreg = episode_lengths["episode_length"]
model = sm.OLS(y_linreg, X_linreg).fit()

# Print summary
print("\n📊 Linear Regression Summary:")
print(model.summary())


# %% 🧮 AGGREGATE FEATURES PER EPISODE  (NO DATA LEAKAGE)
# -----------------------------------------------------------------
# • One row per episode using  mean / std / min / max / q25 / q75
# • Aggregates also the raw episode_length so it becomes episode_length_mean, _std, _min, _max, _q25, _q75            #
# -----------------------------------------------------------------

def aggregate_episode_features(df):
    """Return a DataFrame with one aggregated row per episode, including
       state_, action_ and episode_length_* columns."""
    
    # 1️⃣  flatten per-frame arrays → DataFrames
    state_arr  = np.stack(df["observation.state"].values)
    action_arr = np.stack(df["action"].values)

    state_df  = pd.DataFrame(state_arr,  columns=[f"state_{i}"
                                                  for i in range(state_arr.shape[1])])
    action_df = pd.DataFrame(action_arr, columns=[f"action_{i}"
                                                  for i in range(action_arr.shape[1])])

    # 2️⃣  concat with meta columns and the raw episode_length
    temp = pd.concat(
        [
            df[["task_type", "episode_index", "episode_length"]].reset_index(drop=True),
            state_df,
            action_df
        ],
        axis=1
    )

    # 3️⃣  aggregation plan
    q25 = lambda x: x.quantile(0.25); q25.__name__ = "q25"
    q75 = lambda x: x.quantile(0.75); q75.__name__ = "q75"
    agg_funcs = ["mean", "std", "min", "max", q25, q75]

    feature_cols = (
        state_df.columns.tolist() +
        action_df.columns.tolist() +
        ["episode_length"]                       # ← NEW
    )
    agg_map = {col: agg_funcs for col in feature_cols}

    agg_df = (
        temp.groupby(["task_type", "episode_index"])
            .agg(agg_map)
    )

    # 4️⃣  flatten MultiIndex column names →  state_0_mean … episode_length_q75
    agg_df.columns = [
        f"{col}_{func.__name__ if callable(func) else func}"
        for col, func in agg_df.columns
    ]

    return agg_df.reset_index()

# -----------------------------------------------------------------
# 1️⃣  Aggregate the full DataFrame
full_df = full_df.merge(episode_lengths, on=["task_type", "episode_index"], how="left")
episode_df = aggregate_episode_features(full_df)


# 2️⃣  Numeric task label (0 = box, 1 = drawer)
episode_df["task_label"] = episode_df["task_type"].map({"box": 0, "drawer": 1})

print("✅ Aggregated episode_df shape:", episode_df.shape)

# -----------------------------------------------------------------
# Step 1: Get unique episode indices and their labels (one row per episode)
unique_episodes = episode_df.groupby("episode_index").first().reset_index()
train_ids, test_ids = train_test_split(
    unique_episodes["episode_index"],
    test_size=0.2,
    stratify=unique_episodes["task_label"],
    random_state=SEED
)

# Step 2: Split the full episode_df based on those episode indices
train_ep = episode_df[episode_df["episode_index"].isin(train_ids)].copy()
test_ep = episode_df[episode_df["episode_index"].isin(test_ids)].copy()

# Step 3: Confirm no overlaps
overlap_ids = set(train_ep["episode_index"]).intersection(set(test_ep["episode_index"]))
print(f"🔍 Episode overlap: {len(overlap_ids)}")


########################################

cols_to_drop = ["task_type", "task_label", "task_type_num", "episode_index"]


#############################################

# Separate features and labels
X_train = train_ep.drop(columns=cols_to_drop, errors="ignore")
y_train = train_ep["task_label"]

X_test  = test_ep.drop(columns=cols_to_drop, errors="ignore")
y_test  = test_ep["task_label"]


################################# features with high correlation #########################################


# ==========================================================
# 🔎   Find features that correlate
#      |ρ| ≥ 0.95 with the task label
# ==========================================================
THRESH = 0.95   # tighten / loosen as you like

# 1) build a temp frame: features + label
tmp = (
    pd.concat([pd.DataFrame(X_train, columns=X_train.columns),
               y_train.rename("label")],
              axis=1)
      .select_dtypes(include=[np.number])      # numeric only (safe on old pandas)
)

# 2) compute absolute Pearson correlations with the label
corrs = tmp.corr()["label"].abs()

# 3) grab all features above the threshold
leakers = corrs[corrs >= THRESH].index.drop("label").tolist()
print(f"⚠️  Columns with high correlation with the task label (|ρ| ≥ {THRESH}):\n{leakers}\n")
print(f"Number of highly correlated features: {len(leakers)}")

sns.boxplot(data=pd.DataFrame({leakers[0]: X_train[leakers[0]],
                               "label": y_train.map({0:"box",1:"drawer"})}),
            x="label", y=leakers[0])
plt.title(f"{leakers[0]} vs task label")
plt.show()


# 4) drop them from both train & test
#X_train = X_train.drop(columns=leakers, errors="ignore")
#X_test  = X_test.drop(columns=leakers, errors="ignore")
#print("✅  Shapes after drop:", X_train.shape, X_test.shape)

################################# test #########################################




# %% 🔄 5-FOLD GROUPED CV (ROC-AUC) – ALL SIX MODELS
# --- build *unfitted* pipelines / estimators -----------------------------
cv_model_specs = {
    "Logistic Regression":
        make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=1000, random_state=SEED)),

    "Lasso Logistic Regression":
        make_pipeline(StandardScaler(),
                      LogisticRegressionCV(Cs=10, cv=5, penalty="l1",
                                           solver="saga", scoring="roc_auc",
                                           max_iter=2000, random_state=SEED)),
    "SVM (Linear)":
        make_pipeline(StandardScaler(),
                      SVC(kernel="linear", probability=True,
                          random_state=SEED)),

    # tree models don’t need scaling
    "Random Forest":
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               n_jobs=-1, random_state=SEED),

    "Gradient Boosting":
        GradientBoostingClassifier(random_state=SEED),
}

# --- grouped CV so no episode leaks --------------------------------------
gkf    = GroupKFold(n_splits=5)
groups = train_ep["episode_index"]          # keep full episodes intact

print("📊 5-fold Grouped CV (metric = ROC-AUC)")
for name, est in cv_model_specs.items():
    scores = cross_val_score(
        est, X_train, y_train,
        cv=gkf.split(X_train, y_train, groups=groups),
        scoring="roc_auc", n_jobs=-1
    )
    print(f"{name:25s}  AUC = {scores.mean():.3f}  ± {scores.std():.3f}   {scores}")



# %% 🤖 MODEL TRAINING: Logistic, Lasso, LDA, Random Forest, Gradient Boosting, SVM
# -----------------------------------------------------------------

# ── 1. scale inputs for the linear models
from sklearn.preprocessing import StandardScaler

# 1️⃣ Fit the scaler on *training* data
scaler = StandardScaler()
scaler.fit(X_train)                 # μ and σ come only from X_train

# 2️⃣ Transform train and any future / test data with the SAME scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # ← uses μ and σ learned above

# ── 2. fit the 5 models ---------------------------------------------------
models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000, random_state=SEED).fit(X_train_scaled, y_train),

    "Lasso Logistic Regression":
        LogisticRegressionCV(Cs=10, cv=5, penalty="l1", solver="saga",
                             scoring="roc_auc", max_iter=2000,
                             random_state=SEED).fit(X_train_scaled, y_train),

    "Random Forest":
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               n_jobs=-1, random_state=SEED
            ).fit(X_train, y_train),          # ⬅️ uses raw matrix

    "Gradient Boosting":
        GradientBoostingClassifier(random_state=SEED
            ).fit(X_train, y_train),            # ⬅️ uses raw matrix
        
    "SVM (Linear)":
        SVC(kernel="linear", probability=True, random_state=SEED
            ).fit(X_train_scaled, y_train),
}


# models need the scaled matrix?
SCALED_MODELS = {"Logistic Regression", "Lasso Logistic Regression",
                 "LDA", "SVM (Linear)"}

# %% 📊 EVALUATE ALL MODELS
for name, mdl in models.items():
    X_eval = X_test_scaled if name in SCALED_MODELS else X_test
    y_pred = mdl.predict(X_eval)
    y_prob = mdl.predict_proba(X_eval)[:, 1]

    print(f"\n🔍 {name}")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("AUC      :", roc_auc_score(y_test, y_prob))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 👉 NEW: detailed precision / recall / F1
    print(classification_report(
        y_test,
        y_pred,
        target_names=["box", "drawer"],  # nicer labels
        digits=3                         # consistent rounding
    ))
    
# %% 📈 ROC CURVES (Before PCA – one plot per model)
for name, mdl in models.items():
    # Decide which test data to use (scaled or not)
    X_eval = X_test_scaled if name in SCALED_MODELS else X_test
    y_prob = mdl.predict_proba(X_eval)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)

    # Create individual plot
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f" ROC Curve – {name} (Before PCA)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %% 📊 CONFUSION MATRICES ─ BEFORE PCA  (six models)
for name, mdl in models.items():
    X_eval = X_test_scaled if name in SCALED_MODELS else X_test
    cm     = confusion_matrix(y_test, mdl.predict(X_eval))

    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm,
                annot=True, fmt="d", cbar=False, cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    plt.title(f"{name} – Confusion Matrix (Before PCA)")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.show()

# %% 🔍 FEATURE IMPORTANCES (Coeff / Permutation / MDI)
# ----------------------------------------------------

metric_lbl = {
    "Logistic Regression"       : "|β|  (coefficient)",
    "Lasso Logistic Regression" : "|β|  (coefficient)",
    "SVM (Linear)"              : "|w|  (coefficient)",
    "Random Forest"             : "MDI (Gini importance)",
    "Gradient Boosting"         : "MDI (Gini importance)"
}

TOP_N = 10
fi_long = []

# ── coefficient-based (Logistic, Lasso)
for name in ["Logistic Regression", "Lasso Logistic Regression"]:
    coefs = models[name].coef_[0]
    imp = pd.Series(np.abs(coefs), index=X_train.columns, name="Importance")
    fi_long.append(
        imp.nlargest(TOP_N).reset_index()
           .rename(columns={"index": "Feature"})
           .assign(Model=name, Kind="|β| (coefficient)")
    )

# ── linear SVM coefficients (like logistic)
svm_coefs = np.abs(models["SVM (Linear)"].coef_[0])
imp = pd.Series(svm_coefs, index=X_train.columns, name="Importance")
fi_long.append(
    imp.nlargest(TOP_N).reset_index()
       .rename(columns={"index": "Feature"})
       .assign(Model="SVM (Linear)", Kind="|w| (coefficient)")
)

# ── Gini importance for tree models (RF, GB)
for name in ["Random Forest", "Gradient Boosting"]:
    imp = pd.Series(models[name].feature_importances_,
                    index=X_train.columns, name="Importance")
    fi_long.append(
        imp.nlargest(TOP_N).reset_index()
           .rename(columns={"index": "Feature"})
           .assign(Model=name, Kind="MDI (Gini)")
    )

# ── combine & display
fi_df = pd.concat(fi_long, ignore_index=True)

for mdl in fi_df["Model"].unique():
    sub = fi_df[fi_df["Model"] == mdl].reset_index(drop=True)
    print(f"\n🔹 {mdl} – Top {TOP_N} Features [{metric_lbl[mdl]}]")
    print(sub[["Feature", "Importance"]].to_string(index=False))

    plt.figure(figsize=(8, 5))
    sns.barplot(data=sub, x="Importance", y="Feature", color="steelblue")
    plt.title(f"{mdl} – Top {TOP_N} Feature Importances")
    plt.xlabel("Importance"); plt.tight_layout(); plt.show()

# %% 🔄 Lasso Logistic Regression – BEFORE PCA  (raw episode features)

from sklearn.linear_model import LogisticRegressionCV

# 1️⃣  Fit L1-regularised multinomial logistic regression on the *scaled* matrix
lasso_raw = LogisticRegressionCV(
    penalty="l1",
    solver="saga",              # supports L1 + multinomial
    multi_class="multinomial",
    Cs=np.logspace(-4, 1, 20),  # wider C grid (optional)
    cv=5,
    scoring="roc_auc",          # or "accuracy"
    max_iter=5000,
    random_state=SEED
)
lasso_raw.fit(X_train_scaled, y_train)

# 2️⃣  Plot the cross-validation curve
plot_cv_curve(
    lasso_raw,
    metric_label="Mean CV ROC-AUC",
    title="Lasso Logistic Regression – Before PCA"
)

# 3️⃣  Quick sanity check on the held-out test set
y_prob_raw = lasso_raw.predict_proba(X_test_scaled)[:, 1]
print(f"Test AUC (raw features): {roc_auc_score(y_test, y_prob_raw):.3f}")

# %% 📉 Log-Loss Curve – BEFORE PCA for Gradient Boosting
from sklearn.metrics import log_loss  # place near top if not already there

# Use the model from the dictionary
gb_before = models["Gradient Boosting"]

# Compute staged log-loss for train/test
train_loss = [
    log_loss(y_train, y_pred)
    for y_pred in gb_before.staged_predict_proba(X_train)
]
test_loss = [
    log_loss(y_test, y_pred)
    for y_pred in gb_before.staged_predict_proba(X_test)
]

# Plot
plt.figure(figsize=(6, 4))
plt.plot(train_loss, label="Train")
plt.plot(test_loss,  label="Test")
plt.xlabel("Number of Trees")
plt.ylabel("Log-Loss (Deviance)")
plt.title("Gradient Boosting – BEFORE PCA")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()


# %% 🧠 PCA – Dimensionality Reduction  &  Same 5 Models
# ------------------------------------------------------
# • Fit PCA on *training* data (retain 95 % variance)
# • Evaluate & draw ROC curves
# ------------------------------------------------------

# --- 1️⃣ Fit / transform ----------------------------------------------------
pca = PCA(n_components=0.95, random_state=SEED)
X_train_pca = pca.fit_transform(X_train_scaled)   # fit on train
X_test_pca  = pca.transform(X_test_scaled)        # apply to test

print(f"✅ PCA kept {X_train_pca.shape[1]} components "
      f"({pca.explained_variance_ratio_.sum():.2%} variance)")

# Optional: cumulative variance plot
cum_var = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(7,4))
plt.plot(range(1, len(cum_var)+1), cum_var, marker='o')
plt.axhline(0.95, ls='--', c='red', label='95 %')
plt.xlabel("Number of PCs"); plt.ylabel("Cumulative Variance")
plt.title("PCA Cumulative Explained Variance"); plt.grid(); plt.legend(); plt.tight_layout(); plt.show()

# --- 2️⃣ (re-)train the SIX models ––––––––––––––––––––––––––––––––––––––
from sklearn.ensemble import RandomForestClassifier    # <- new import (put with the others)

# 1️⃣ create UNFITTED estimators
models_pca = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000, random_state=SEED),

    "Lasso Logistic Regression":
        LogisticRegressionCV(Cs=10, cv=5, penalty="l1", solver="saga",
                             scoring="roc_auc", max_iter=2000,
                             random_state=SEED),

    "Random Forest":
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               n_jobs=-1, random_state=SEED),

    "Gradient Boosting":
        GradientBoostingClassifier(random_state=SEED),

    "SVM (Linear)":
        SVC(kernel="linear", probability=True, random_state=SEED),
}

# 2️⃣ train each one ONCE on the PCA features
for name, mdl in models_pca.items():
    mdl.fit(X_train_pca, y_train)

# %% 📉 Log-Loss Curve – AFTER PCA for Gradient Boosting
gb_after = models_pca["Gradient Boosting"]

train_loss_pca = [
    log_loss(y_train, y_pred)
    for y_pred in gb_after.staged_predict_proba(X_train_pca)
]
test_loss_pca = [
    log_loss(y_test, y_pred)
    for y_pred in gb_after.staged_predict_proba(X_test_pca)
]

# Plot
plt.figure(figsize=(6, 4))
plt.plot(train_loss_pca, label="Train")
plt.plot(test_loss_pca,  label="Test")
plt.xlabel("Number of Trees")
plt.ylabel("Log-Loss (Deviance)")
plt.title("Gradient Boosting – AFTER PCA")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# --- 3️⃣ evaluate ––––––––––––––––––––––––––––––––––––––––––––––––––––
print("\n📊 PCA-MODEL PERFORMANCE")
for name, mdl in models_pca.items():
    y_pred = mdl.predict(X_test_pca)
    y_prob = mdl.predict_proba(X_test_pca)[:, 1]

    print(f"\n🔍 {name}")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("AUC      :", roc_auc_score(y_test, y_prob))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 👉 NEW: classification report
    print(classification_report(
        y_test,
        y_pred,
        target_names=["box", "drawer"],
        digits=3
    ))

# --- 4️⃣ ROC curves ––––––––––––––––––––––––––––––––––––––––––––––––––
for name, mdl in models_pca.items():
    y_prob = mdl.predict_proba(X_test_pca)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"ROC Curve – {name} (After PCA)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# %% 📊 CONFUSION MATRIX – **AFTER PCA**
for name, mdl in models_pca.items():
    cm = confusion_matrix(y_test, mdl.predict(X_test_pca))

    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm,
                annot=True, fmt="d", cbar=False, cmap="Greens",
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    plt.title(f"{name} – Confusion Matrix (After PCA)")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.show()
    
# %% 🔍 FEATURE IMPORTANCE  (after PCA)  
# -------------------------------------------------------------------
TOP_N = 10
pc_cols = [f"PC{i+1}" for i in range(X_train_pca.shape[1])]
fi_pca  = []

# ① coefficient-based (Logistic, Lasso, *linear SVM*)
for name in ["Logistic Regression",
             "Lasso Logistic Regression",
             "SVM (Linear)"]:
    coefs = models_pca[name].coef_[0]
    s = pd.Series(np.abs(coefs), index=pc_cols, name="Importance")
    fi_pca.append(
        s.nlargest(TOP_N).reset_index()
          .rename(columns={"index": "Feature"})
          .assign(Model=name, Kind="|β| (coefficient)")
    )

# ③ built-in impurity-based importances (RF & GB)
for tree_name in ["Random Forest", "Gradient Boosting"]:
    imp = pd.Series(models_pca[tree_name].feature_importances_,
                    index=pc_cols, name="Importance")
    fi_pca.append(
        imp.nlargest(TOP_N).reset_index()
           .rename(columns={"index": "Feature"})
           .assign(Model=tree_name, Kind="Gini importance")
    )

# --- combine / print / plot -----------------------------------------------
fi_pca_df = pd.concat(fi_pca, ignore_index=True)

for mdl in fi_pca_df["Model"].unique():
    sub = fi_pca_df[fi_pca_df["Model"] == mdl]
    print(f"\n🔹 {mdl} – Top {TOP_N} Features [{sub['Kind'].iloc[0]}]")
    print(sub[["Feature", "Importance"]].to_string(index=False))

    plt.figure(figsize=(7, 4))
    sns.barplot(data=sub, x="Importance", y="Feature", color="steelblue")
    plt.title(f"{mdl} – Top {TOP_N} Importances (after PCA)")
    plt.tight_layout(); plt.show()



# %%  🔄 Lasso Logistic Regression – AFTER PCA  
pca = PCA(n_components=0.95, random_state=SEED)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

lasso_pca = LogisticRegressionCV(
    penalty="l1", solver="saga",
    Cs=10, cv=5, scoring="roc_auc",
    max_iter=2000, random_state=SEED
)
lasso_pca.fit(X_train_pca, y_train)

plot_cv_curve(lasso_pca,
              metric_label="Mean CV ROC-AUC",
              title="Lasso Logistic Regression – After PCA")
######################################################################

# %% 📉 Log-Loss Curve – AFTER PCA for Gradient Boosting
# ------------------------------------------------------

# Retrain Gradient Boosting on PCA-transformed data
gb_pca = GradientBoostingClassifier(random_state=SEED)
gb_pca.fit(X_train_pca, y_train)

# Collect staged log-losses
train_loss_pca = [
    log_loss(y_train, y_pred)
    for y_pred in gb_pca.staged_predict_proba(X_train_pca)
]
test_loss_pca = [
    log_loss(y_test, y_pred)
    for y_pred in gb_pca.staged_predict_proba(X_test_pca)
]

# Plot the log-loss curve
plt.figure(figsize=(6, 4))
plt.plot(train_loss_pca, label="Train")
plt.plot(test_loss_pca,  label="Test")
plt.xlabel("Number of Trees")
plt.ylabel("Log-Loss (Deviance)")
plt.title("Gradient Boosting – AFTER PCA")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %% 📊 Visualize PC1 vs Task Label

# Create a DataFrame for visualization
viz_df = pd.DataFrame({
    "PC1": X_train_pca[:, 0],                # first principal component
    "task_label": y_train.map({0: "box", 1: "drawer"})  # human-readable labels
})

# Boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(data=viz_df, x="task_label", y="PC1", palette="Set2")
plt.title("Distribution of PC1 by Task Type")
plt.xlabel("Task Type")
plt.ylabel("PC1 Value")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% 📋 Print All Features Contributing to PC1 (Sorted by Absolute Value)

# Get feature names from original data
original_features = X_train.columns

# Get PC1 loadings
pc1_loadings = pd.Series(
    data=pca.components_[0],     # PC1 is the first row of components_
    index=original_features,
    name="Loading"
)

# Sort by absolute value of the loadings
pc1_sorted = pc1_loadings.reindex(pc1_loadings.abs().sort_values(ascending=False).index)

# Print top features
print("🔍 Features contributing most to PC1 (sorted by |loading|):")
print(pc1_sorted.to_string())

# %% 📊 Top-20 contributors to PC 1  ─ ordered by |loading|

TOP_N = 20

# 1. Rank by absolute value of the loading
pc1_top = (
    pc1_loadings
      .abs()                                  # strength
      .sort_values(ascending=False)           # biggest → smallest
      .head(TOP_N)
)

# 2. Build a DataFrame that keeps both sign & strength
df_pc1 = (
    pd.DataFrame({
        "Feature" : pc1_top.index,
        "Loading" : pc1_loadings.loc[pc1_top.index],  # signed
        "Strength": pc1_top.values                    # |loading|
    })
)

# 3. Plot (largest strength at the top)
plt.figure(figsize=(8, 6))
sns.barplot(
    data=df_pc1,
    y="Feature",
    x="Loading",           # signed value for direction
    palette="vlag",
    orient="h"
)
plt.axvline(0, color="k", lw=1)
plt.title(f"Top {TOP_N} Features Contributing to PC1")
plt.xlabel("PC1 Loading (sign shows direction, length shows strength)")
plt.ylabel("Original Feature")
plt.gca().invert_yaxis()   # biggest strength at the top
plt.tight_layout()
plt.show()

# %% 📊 PC1 – State vs Action:  Count **and** Strength  🔍

# ---------- 1) absolute loadings for PC1 -------------
pc1_abs = pc1_loadings.abs()          # already defined earlier

# ---------- 2) tag every feature ---------------------
tags = pc1_abs.index.to_series().str.startswith("state_")
source = tags.map({True: "state", False: "action"})

# ---------- 3) aggregate ------------------------------
count_per_src = source.value_counts().reindex(["state", "action"], fill_value=0)
sumabs_per_src = pc1_abs.groupby(source).sum().reindex(["state", "action"], fill_value=0)

# ---------- 4) build a tidy DataFrame ----------------
df_plot = (pd.DataFrame({
                "Count": count_per_src,
                "Total |loading|": sumabs_per_src
            })
            .reset_index()
            .rename(columns={"index": "Source"}))

# ---------- 5) plot -----------------------------------
fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=False)

# Counts
sns.barplot(data=df_plot, x="Source", y="Count",
            palette=["#4C72B0", "#55A868"], ax=ax[0])
ax[0].set_title("PC1 – Number of Contributing Features")
ax[0].set_ylabel("Count")
sns.barplot(data=df_plot, x="Source", y="Total |loading|",
            palette=["#4C72B0", "#55A868"], ax=ax[1])
ax[1].set_title("PC1 – Sum of |loadings|\n(stronger = more influence)")
ax[1].set_ylabel("Sum of |loadings|  (stronger = more influence)")

for a in ax:
    a.set_xlabel(""); a.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

# ---------- 6) print exact numbers --------------------
print("📊 PC1 – Source breakdown")
print(df_plot.set_index("Source").round(4))


