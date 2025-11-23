# %%
import warnings
warnings.filterwarnings("ignore")

# ===========================================
# IMPORT LIBRARIES
# ===========================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================================
# LOAD DATA (VS CODE — NO COLAB FILE UPLOAD)
# ===========================================
df = pd.read_csv(r"C:\Users\saksh\OneDrive\Desktop\poster\air_quality_health_impact_data.csv")

# %%
print("Shape of dataset:", df.shape)
print("\n--- Basic Info ---")
df.info()

print("\n--- Summary Statistics ---")
print(df.describe())

print("\nFirst 5 rows:")
print(df.head())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# %%
# ===========================================
# TARGET VARIABLE CHECK
# ===========================================
target = "HealthImpactClass"

if target in df.columns:
    print(f"\n--- Target Variable '{target}' Distribution ---")
    print(df[target].value_counts())

    plt.figure(figsize=(6,4))
    sns.countplot(x=target, data=df, palette="viridis")
    plt.title(f"Distribution of Target: {target}")
    plt.show()
else:
    print(f"Target column '{target}' not found!")

# %%
# ===========================================
# CORRELATION HEATMAP
# ===========================================
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.3)
plt.title("Correlation Heatmap")
plt.show()

# %%
# ===========================================
# HISTOGRAMS
# ===========================================
df.hist(figsize=(12,8), bins=15, color='skyblue', edgecolor='black')
plt.suptitle("Histograms for Numeric Columns")
plt.show()

# %%
# ===========================================
# BOX PLOTS
# ===========================================
numeric_cols = df.select_dtypes(include=['int64','float64']).columns

for col in numeric_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[col], color='skyblue')
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.show()

# %%
# ===========================================
# PAIRPLOT (first 5 numeric features)
# ===========================================
sns.pairplot(df[numeric_cols[:5]], diag_kind='kde', corner=True)
plt.suptitle("Pairplot (First 5 Numeric Features)", y=1.02)
plt.show()

# %%
# ===========================================
# SKEWNESS
# ===========================================
print("\n--- Skewness of Numeric Columns ---")
print(df[numeric_cols].skew())

# %%
# ===========================================
# CORRELATION WITH TARGET
# ===========================================
if target in df.columns:
    numeric_features = df.select_dtypes(include=['int64','float64']).columns
    numeric_features = numeric_features.difference([target])  # Safe drop

    correlations = df[numeric_features].corrwith(df[target])
    print("\n--- Correlation of Features with Target ---")
    print(correlations.sort_values(ascending=False))

    plt.figure(figsize=(8,5))
    correlations.sort_values().plot(kind='barh', color='teal')
    plt.title(f"Correlation of Features with {target}")
    plt.xlabel("Correlation Coefficient")
    plt.show()

print("\nEDA Completed Successfully!")

# %%
# ===========================================
# OUTLIER CAPPING
# ===========================================
plt.figure(figsize=(8,4))
sns.boxplot(x=df['HealthImpactScore'], color='skyblue')
plt.title("Before Capping: HealthImpactScore")
plt.show()

Q1 = df['HealthImpactScore'].quantile(0.25)
Q3 = df['HealthImpactScore'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

df['HealthImpactScore'] = df['HealthImpactScore'].clip(lower=lower_limit, upper=upper_limit)

plt.figure(figsize=(8,4))
sns.boxplot(x=df['HealthImpactScore'], color='lightgreen')
plt.title("After Capping: HealthImpactScore")
plt.show()

# %%
# ===========================================
# MACHINE LEARNING MODELS
# ===========================================
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import xgboost

# %%
# ===========================================
# TRAIN TEST SPLIT
# ===========================================
X = df.drop(columns=[target])
y = df[target]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# ===========================================
# LOGISTIC REGRESSION
# ===========================================
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("\nLogistic Regression Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)).plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

# %%
# ===========================================
# DECISION TREE + GRID SEARCH
# ===========================================
dt_params = {'criterion': ['gini','entropy'], 'max_depth': [3,5,10,None], 'min_samples_split': [2,5,10]}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5, scoring='accuracy')
dt_grid.fit(X_train, y_train)
best_dt = dt_grid.best_estimator_
y_pred = best_dt.predict(X_test)

print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)).plot()
plt.title("Decision Tree")
plt.show()

# %%
# ===========================================
# RANDOM FOREST
# ===========================================
rf_grid = {'n_estimators':[100,200], 'max_depth':[None,10,20], 'min_samples_split':[2,5,10], 'criterion':['gini','entropy']}
rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf.fit(X_train_scaled, y_train)
best_rf = rf.best_estimator_
y_pred = best_rf.predict(X_test_scaled)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred))

# %%
# ===========================================
# KNN
# ===========================================
knn_grid = {'n_neighbors':[3,5,7,9], 'weights':['uniform','distance'], 'metric':['euclidean','manhattan']}
knn = GridSearchCV(KNeighborsClassifier(), knn_grid, cv=5, scoring='accuracy', n_jobs=-1)
knn.fit(X_train_scaled, y_train)
best_knn = knn.best_estimator_
y_pred = best_knn.predict(X_test_scaled)
print("\nKNN Accuracy:", accuracy_score(y_test, y_pred))

# %%
# ===========================================
# SVM
# ===========================================
svm = SVC(probability=True, class_weight='balanced', random_state=42)
svm_grid = {'C':[0.1,1,10], 'kernel':['linear','rbf','poly'], 'gamma':['scale','auto']}
svm_search = GridSearchCV(svm, svm_grid, cv=5, scoring='accuracy', n_jobs=-1)
svm_search.fit(X_train_scaled, y_train)
best_svm = svm_search.best_estimator_
y_pred = best_svm.predict(X_test_scaled)
print("\nSVM Accuracy:", accuracy_score(y_test, y_pred))

# %%
# ===========================================
# ADA BOOST
# ===========================================
ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
ada.fit(X_train_scaled, y_train)
y_pred = ada.predict(X_test_scaled)
print("\nAdaBoost Accuracy:", accuracy_score(y_test, y_pred))

# %%
# ===========================================
# XGBOOST
# ===========================================
print("XGBoost version:", xgboost.__version__)
xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='mlogloss')
xgb.fit(X_train_scaled, y_train)
y_pred = xgb.predict(X_test_scaled)
print("\nXGBoost Accuracy:", accuracy_score(y_test, y_pred))

# %%
# ===========================================
# MODEL COMPARISON PLOT (DYNAMIC)
# ===========================================
model_names = ["Logistic Regression", "Decision Tree", "Random Forest", "KNN", "SVM", "AdaBoost", "XGBoost"]
accuracies = [
    accuracy_score(y_test, model.predict(X_test_scaled)),         # Logistic Regression
    accuracy_score(y_test, best_dt.predict(X_test)),              # Decision Tree
    accuracy_score(y_test, best_rf.predict(X_test_scaled)),       # Random Forest
    accuracy_score(y_test, best_knn.predict(X_test_scaled)),      # KNN
    accuracy_score(y_test, best_svm.predict(X_test_scaled)),      # SVM
    accuracy_score(y_test, ada.predict(X_test_scaled)),           # AdaBoost
    accuracy_score(y_test, xgb.predict(X_test_scaled))            # XGBoost
]

df_results = pd.DataFrame({'Model': model_names, 'Accuracy': accuracies})
df_results = df_results.sort_values(by='Accuracy', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Accuracy', y='Model', data=df_results, palette='coolwarm')
plt.title("Model Accuracy Comparison")
plt.show()

#%%
import joblib
# Example: saving XGBoost model
joblib.dump(xgb, "xgb_model.pkl")