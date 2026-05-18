"""
============================================================
 Farmer Credit Risk Classifier — Kabacan, North Cotabato
 Course: Computational Science for Computer Science
 Algorithm: ID3 Decision Tree (entropy criterion, max_depth=4)
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)
from sklearn.preprocessing import LabelEncoder

# ── 0. Reproducibility ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── 1. Load Dataset ───────────────────────────────────────────────────────────
print("=" * 60)
print(" FARMER CREDIT RISK CLASSIFIER — KABACAN, NORTH COTABATO")
print("=" * 60)

df = pd.read_csv("farmer credit risk.csv")
print(f"\n[Dataset] Shape: {df.shape}")
print(f"[Dataset] Class distribution:\n{df['credit_risk'].value_counts()}\n")

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
# One-hot encode crop_type (categorical)
df_encoded = pd.get_dummies(df, columns=["crop_type"], drop_first=False)

# Encode target: High=1, Low=0
le = LabelEncoder()
y = le.fit_transform(df_encoded["credit_risk"])   # High→0, Low→1 (alphabetical)
# Remap so High=1, Low=0 for clarity
label_map = {cls: i for i, cls in enumerate(le.classes_)}
print(f"[Encoding] Label map: {label_map}")

X = df_encoded.drop(columns=["credit_risk"])
feature_names = X.columns.tolist()

print(f"[Preprocessing] Features after encoding: {len(feature_names)}")
print(f"[Preprocessing] Feature list: {feature_names}\n")

# ── 3. Train / Test Split (80/20, stratified) ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
print(f"[Split] Training set: {X_train.shape[0]} records")
print(f"[Split] Testing set:  {X_test.shape[0]} records\n")

# ── 4. ID3 Decision Tree — Baseline (max_depth=4, entropy) ───────────────────
print("-" * 60)
print(" MODEL: ID3 Decision Tree  |  criterion=entropy  |  max_depth=4")
print("-" * 60)

id3 = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=SEED)
id3.fit(X_train, y_train)

y_pred = id3.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n[Result] Test Accuracy:  {acc * 100:.2f}%")
print("\n[Result] Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── 5. Confusion Matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
print("[Confusion Matrix]")
print(f"  Actual \\ Predicted  |  {le.classes_[0]}  |  {le.classes_[1]}")
print(f"  {le.classes_[0]:18s}|  {cm[0][0]:4d}    |  {cm[0][1]:4d}")
print(f"  {le.classes_[1]:18s}|  {cm[1][0]:4d}    |  {cm[1][1]:4d}\n")

# ── 6. Experiment A — Tree Depth Control ─────────────────────────────────────
print("=" * 60)
print(" EXPERIMENT A — Tree Depth Control")
print("=" * 60)

depths      = [2, 3, 4, 5]
train_accs  = []
test_accs   = []
leaf_counts = []
node_counts = []

for d in depths:
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=SEED)
    clf.fit(X_train, y_train)
    tr_acc = accuracy_score(y_train, clf.predict(X_train))
    te_acc = accuracy_score(y_test,  clf.predict(X_test))
    train_accs.append(tr_acc * 100)
    test_accs.append(te_acc  * 100)
    leaf_counts.append(clf.get_n_leaves())
    node_counts.append(clf.tree_.node_count)
    marker = " ← SELECTED" if d == 4 else ""
    print(f"  depth={d}  |  Train: {tr_acc*100:.2f}%  |  Test: {te_acc*100:.2f}%  "
          f"|  Leaves: {clf.get_n_leaves()}  |  Nodes: {clf.tree_.node_count}{marker}")

# ── 7. Experiment B — Cross-Validation Stability ─────────────────────────────
print("\n" + "=" * 60)
print(" EXPERIMENT B — Cross-Validation Stability")
print("=" * 60)

id3_d4 = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=SEED)

cv5  = StratifiedKFold(n_splits=5,  shuffle=True, random_state=SEED)
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

scores_5  = cross_val_score(id3_d4, X, y, cv=cv5,  scoring="accuracy")
scores_10 = cross_val_score(id3_d4, X, y, cv=cv10, scoring="accuracy")

print(f"\n  80/20 Split       |  Accuracy: {acc*100:.2f}%  |  Std: N/A")
print(f"  5-Fold  CV        |  Mean: {scores_5.mean()*100:.2f}%  |  "
      f"Std: ±{scores_5.std()*100:.2f}%")
print(f"  10-Fold CV        |  Mean: {scores_10.mean()*100:.2f}%  |  "
      f"Std: ±{scores_10.std()*100:.2f}%")

# ── 8. Experiment C — Feature Importance ─────────────────────────────────────
print("\n" + "=" * 60)
print(" EXPERIMENT C — Feature Importance (Mean Decrease in Impurity)")
print("=" * 60)

importances = id3.feature_importances_
importance_df = pd.DataFrame({
    "Feature":    feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print()
for _, row in importance_df.iterrows():
    bar = "█" * int(row["Importance"] * 50)
    print(f"  {row['Feature']:30s}  {row['Importance']:.4f} ({row['Importance']*100:.1f}%)  {bar}")

# ── 9. Experiment D — Entropy vs Gini ────────────────────────────────────────
print("\n" + "=" * 60)
print(" EXPERIMENT D — Entropy (ID3) vs Gini Impurity (CART)")
print("=" * 60)

gini = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=SEED)
gini.fit(X_train, y_train)
gini_acc    = accuracy_score(y_test, gini.predict(X_test))
gini_cv5    = cross_val_score(gini, X, y, cv=cv5, scoring="accuracy")

print(f"\n  Entropy (ID3)  |  Test: {acc*100:.2f}%  |  "
      f"5-Fold CV: {scores_5.mean()*100:.2f}% (±{scores_5.std()*100:.2f}%)")
print(f"  Gini (CART)    |  Test: {gini_acc*100:.2f}%  |  "
      f"5-Fold CV: {gini_cv5.mean()*100:.2f}% (±{gini_cv5.std()*100:.2f}%)")

# ── 10. Information Gain Verification (Root Level) ───────────────────────────
print("\n" + "=" * 60)
print(" MANUAL INFORMATION GAIN VERIFICATION — Root Level")
print("=" * 60)

def entropy(y_arr):
    classes, counts = np.unique(y_arr, return_counts=True)
    probs = counts / len(y_arr)
    return -np.sum(probs * np.log2(probs + 1e-12))

def information_gain(X_col, y_arr, threshold=None):
    h_parent = entropy(y_arr)
    if threshold is not None:
        left  = y_arr[X_col <= threshold]
        right = y_arr[X_col >  threshold]
        if len(left) == 0 or len(right) == 0:
            return 0
        h_child = (len(left)/len(y_arr))  * entropy(left) + \
                  (len(right)/len(y_arr)) * entropy(right)
    else:
        vals = np.unique(X_col)
        h_child = sum((len(y_arr[X_col == v]) / len(y_arr)) *
                      entropy(y_arr[X_col == v]) for v in vals)
    return h_parent - h_child

y_full  = y
X_full  = X.values
cols    = X.columns.tolist()

print(f"\n  Root entropy H(S) = {entropy(y_full):.4f}\n")
print(f"  {'Feature':<30}  {'IG':>8}  {'Type'}")
print(f"  {'-'*55}")

ig_results = []
for i, col in enumerate(cols):
    col_vals = X_full[:, i]
    unique   = np.unique(col_vals)
    if len(unique) <= 4:   # binary / categorical (post-encoding)
        ig = information_gain(col_vals, y_full)
        ig_results.append((col, ig, "binary/cat"))
    else:                  # continuous — find best threshold
        best_ig, best_t = 0, None
        thresholds = [(unique[j] + unique[j+1]) / 2 for j in range(len(unique)-1)]
        for t in thresholds:
            ig_t = information_gain(col_vals, y_full, threshold=t)
            if ig_t > best_ig:
                best_ig, best_t = ig_t, t
        ig_results.append((col, best_ig, f"cont, t={best_t:.4f}"))

ig_results.sort(key=lambda x: x[1], reverse=True)
for feat, ig, ftype in ig_results:
    print(f"  {feat:<30}  {ig:>8.4f}  {ftype}")

# ── 11. Visualizations ───────────────────────────────────────────────────────
print("\n[Visualizations] Generating plots...")

colors  = {"primary": "#1F4E79", "secondary": "#2E75B6",
           "accent": "#E2EFDA",  "warning": "#C55A11", "light": "#DEEAF1"}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Farmer Credit Risk Classifier — Kabacan, North Cotabato\n"
             "ID3 Decision Tree | Entropy Criterion | max_depth=4",
             fontsize=14, fontweight="bold", color=colors["primary"], y=0.98)

# ── Plot 1: Confusion Matrix ──────────────────────────────────────────────────
ax1 = axes[0, 0]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax1, colorbar=False, cmap="Blues")
ax1.set_title("Confusion Matrix\n(Test Set, n=200)", fontweight="bold",
              color=colors["primary"])
ax1.set_xlabel("Predicted Label", color=colors["primary"])
ax1.set_ylabel("Actual Label",    color=colors["primary"])

# ── Plot 2: Depth Control ─────────────────────────────────────────────────────
ax2 = axes[0, 1]
ax2.plot(depths, train_accs, "o--", color=colors["secondary"],
         label="Train Accuracy", linewidth=2, markersize=8)
ax2.plot(depths, test_accs,  "o-",  color=colors["warning"],
         label="Test Accuracy",  linewidth=2, markersize=8)
ax2.axvline(x=4, color="gray", linestyle=":", linewidth=1.5,
            label="Selected depth=4")
ax2.set_title("Experiment A — Tree Depth Control",
              fontweight="bold", color=colors["primary"])
ax2.set_xlabel("max_depth",   color=colors["primary"])
ax2.set_ylabel("Accuracy (%)", color=colors["primary"])
ax2.legend()
ax2.set_xticks(depths)
ax2.grid(True, alpha=0.3)
for d, tr, te in zip(depths, train_accs, test_accs):
    ax2.annotate(f"{te:.1f}%", (d, te), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=9, color=colors["warning"])

# ── Plot 3: Feature Importance ────────────────────────────────────────────────
ax3 = axes[1, 0]
imp_sorted = importance_df.sort_values("Importance")
bar_colors = [colors["warning"] if imp_sorted["Importance"].iloc[-1] == v
              else colors["secondary"] for v in imp_sorted["Importance"]]
bars = ax3.barh(imp_sorted["Feature"], imp_sorted["Importance"],
                color=bar_colors, edgecolor="white", linewidth=0.5)
ax3.set_title("Experiment C — Feature Importance\n(Mean Decrease in Impurity)",
              fontweight="bold", color=colors["primary"])
ax3.set_xlabel("Importance Score", color=colors["primary"])
ax3.set_xlim(0, imp_sorted["Importance"].max() * 1.2)
for bar, val in zip(bars, imp_sorted["Importance"]):
    ax3.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", fontsize=8)
ax3.grid(True, axis="x", alpha=0.3)

# ── Plot 4: Entropy vs Gini (bar comparison) ──────────────────────────────────
ax4 = axes[1, 1]
criteria    = ["Entropy (ID3)", "Gini (CART)"]
test_scores = [acc * 100, gini_acc * 100]
cv_scores   = [scores_5.mean() * 100, gini_cv5.mean() * 100]
cv_stds     = [scores_5.std()  * 100, gini_cv5.std()  * 100]

x    = np.arange(len(criteria))
w    = 0.35
b1   = ax4.bar(x - w/2, test_scores, w, label="Test Accuracy",
               color=colors["primary"],   edgecolor="white")
b2   = ax4.bar(x + w/2, cv_scores,   w, label="5-Fold CV Accuracy",
               color=colors["secondary"], edgecolor="white",
               yerr=cv_stds, capsize=5, error_kw={"ecolor": colors["warning"]})
ax4.set_title("Experiment D — Entropy vs Gini Impurity\n(depth=4)",
              fontweight="bold", color=colors["primary"])
ax4.set_ylabel("Accuracy (%)", color=colors["primary"])
ax4.set_xticks(x)
ax4.set_xticklabels(criteria)
ax4.set_ylim(70, 90)
ax4.legend()
ax4.grid(True, axis="y", alpha=0.3)
for bar in b1:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
for bar in b2:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("visualization_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("[Visualizations] Saved → visualization_results.png")

# ── 12. Decision Tree Text Export ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(" DECISION TREE STRUCTURE (text export)")
print("=" * 60)
tree_text = export_text(id3, feature_names=feature_names, max_depth=4)
print(tree_text)

# ── 13. Sample Prediction — Loan Officer Demo ─────────────────────────────────
print("=" * 60)
print(" SAMPLE FARMER PROFILE PREDICTION")
print("=" * 60)

sample_farmer = {
    "farm_size_ha":        2.5,
    "years_experience":    8,
    "current_debt_php":    45000,
    "loan_repayment_hist": 0,       # Poor history
    "irrigation_access":   1,       # Has irrigation
    "avg_yield_tons_ha":   3.1,
    "crop_type_Corn":      0,
    "crop_type_Rice":      1,       # Grows rice
    "crop_type_Soybean":   0,
    "crop_type_Vegetable": 0,
}

sample_df   = pd.DataFrame([sample_farmer])[feature_names]
prediction  = id3.predict(sample_df)[0]
label       = le.inverse_transform([prediction])[0]
proba       = id3.predict_proba(sample_df)[0]

print(f"\n  Farm size:            {sample_farmer['farm_size_ha']} ha")
print(f"  Experience:           {sample_farmer['years_experience']} years")
print(f"  Current debt:         ₱{sample_farmer['current_debt_php']:,}")
print(f"  Repayment history:    {'Good' if sample_farmer['loan_repayment_hist'] else 'Poor'}")
print(f"  Irrigation access:    {'Yes' if sample_farmer['irrigation_access'] else 'No'}")
print(f"  Average yield:        {sample_farmer['avg_yield_tons_ha']} tons/ha")
print(f"  Crop type:            Rice")
print(f"\n  ► Predicted Credit Risk: {label.upper()}")
for cls, p in zip(le.classes_, proba):
    print(f"    {cls}: {p*100:.1f}%")

print("\n[Done] All experiments complete.")