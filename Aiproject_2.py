import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer

# Load dataset
df = pd.read_csv('heart_disease_uci.csv', index_col=0)
df.columns = ['age', 'gender', 'dataset', 'cp', 'trestbps', 'chol', 'fbs',
              'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

# Downcasting Err
pd.set_option('future.no_silent_downcasting', True)

# Handle missing values
numeric_columns = ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
categorical_columns = ['gender', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode().iloc[0])


for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())



df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

cp_mapping = {
    'typical angina': 0,
    'atypical angina': 1,
    'non-anginal': 2,
    'asymptomatic': 3
}
df['cp'] = df['cp'].map(cp_mapping)
df['fbs'] = df['fbs'].map({True: 1, False: 0})

restecg_mapping = {
    'normal': 0,
    'st-t abnormality': 1,
    'lv hypertrophy': 2
}
df['restecg'] = df['restecg'].map(restecg_mapping)
df['exang'] = df['exang'].map({True: 1, False: 0})
slope_mapping = {
    'upsloping': 0,
    'flat': 1,
    'downsloping': 2
}
df['slope'] = df['slope'].map(slope_mapping)
thal_mapping = {
    'normal': 2,
    'fixed defect': 1,
    'reversable defect': 3
}
df['thal'] = df['thal'].map(thal_mapping)

df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

# Remove duplicates
df = df.drop_duplicates(keep='last').reset_index(drop=True)

# Outlier Removal using Z-scores
z_scores = stats.zscore(df[numeric_columns])
outliers = (np.abs(z_scores) > 3).any(axis=1)
df = df[~outliers].reset_index(drop=True)

# Label Encoding for categorical variables
encoder = LabelEncoder()
'''
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])
'''
df['dataset'] = encoder.fit_transform(df['dataset'])

# Scaling numeric features
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Feature Engineering
df['ageGroup'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 70, 90],
                         labels=['<30', '30-40', '40-50', '50-60', '60-70', '>70'])
df['ageGroup'] = encoder.fit_transform(df['ageGroup'].astype(str))
df['heartRateReserve'] = 220 - df['age'] - df['thalch']
df['cholFbsInteraction'] = df['chol'] * df['fbs']
new_numeric_columns = ['heartRateReserve', 'cholFbsInteraction']
df[new_numeric_columns] = scaler.fit_transform(df[new_numeric_columns])


# Correlation matrix and heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Target and features
target = df['num']
features = df.drop(columns=['num'])

# Check fo class imbalance
print(target.value_counts())
print(target.value_counts(normalize=True))

# class distribution
plt.figure(figsize=(10, 8))
target.value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

imbalance_techniques = {
    'SMOTE': SMOTE(random_state=42),
    'Random Over-sampling': RandomOverSampler(random_state=42),
    'Random Under-sampling': RandomUnderSampler(random_state=42)
}

for name, technique in imbalance_techniques.items():
    print(f"\n{name} Resampling:")
    X_resampled, y_resampled = technique.fit_resample(features, target)
    plt.figure(figsize=(10, 8))
    pd.Series(y_resampled).value_counts().plot(kind='bar')
    plt.title(f'{name} Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    print("Resampled Class Distribution:")
    print(pd.Series(y_resampled).value_counts())

plt.show()

# PCA Transform
pca = PCA(n_components=5)
pca_feat = pca.fit_transform(features)

# PCA Heatmap
pca_df = pd.DataFrame(pca_feat, columns=[f'PC{i+1}' for i in range(5)])
plt.figure(figsize=(10, 8))
sns.heatmap(pca_df.corr(), annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix of PCA')
plt.show()

################Phase_2################

X = features
Y = target

def train_evaluate_supervised_models(X, y, with_pca=False):

    hastech = False

    models = {
        'SVM': SVC(kernel='rbf', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    results = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    if with_pca:
        pca = PCA(n_components=5)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)


    for model_name, model in models.items():

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        results[model_name] = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix
        }

        print(f"\n{model_name}:")
        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:")
        print(conf_matrix)

    return results , hastech

# Run supervised models without and with PCA

# best method Logistic Regression without PCA

# next step with balancing

def evaluate_models_with_balancing(X, y):

    hastech = True

    balancing_techniques = {
        'Unbalanced': None,
        'SMOTE': SMOTE(random_state=42),
        'Random Over-sampling': RandomOverSampler(random_state=42),
        'Random Under-sampling': RandomUnderSampler(random_state=42)
    }

    models = {
        'SVM': SVC(kernel='rbf', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }


    results = {}

    for technique_name, balancing_technique in balancing_techniques.items():
        print(f"\n--- {technique_name} Technique ---")


        if balancing_technique:
            X_resampled, y_resampled = balancing_technique.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y


        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        results[technique_name] = {}

        for model_name, model in models.items():

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            results[technique_name][model_name] = {
                'accuracy': accuracy,
                'confusion_matrix': conf_matrix
            }

            print(f"\n{model_name}:")
            print(f"Accuracy: {accuracy}")
            print("Confusion Matrix:")
            print(conf_matrix)


    return results , hastech

# With balancing and pcaa
def evaluate_models_with_pca_and_balancing(X, y, n_components=5):

    hastech = True

    balancing_techniques = {
        'Unbalanced': None,
        'SMOTE': SMOTE(random_state=42),
        'Random Over-sampling': RandomOverSampler(random_state=42),
        'Random Under-sampling': RandomUnderSampler(random_state=42)
    }

    models = {
        'SVM': SVC(kernel='rbf', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    results = {}

    for technique_name, balancing_technique in balancing_techniques.items():
        print(f"\n--- {technique_name} Technique with PCA ---")

        if balancing_technique:
            X_resampled, y_resampled = balancing_technique.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y

        pca = PCA(n_components=n_components)
        X_resampled_pca = pca.fit_transform(X_resampled)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled_pca, y_resampled, test_size=0.2, random_state=42)

        results[technique_name] = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            results[technique_name][model_name] = {
                'accuracy': accuracy,
                'confusion_matrix': conf_matrix
            }

            # Print results
            print(f"\n{model_name}:")
            print(f"Accuracy: {accuracy}")
            print(f"Confusion Matrix:\n{conf_matrix}")

    return results, hastech


def plot_accuracy_comparison(results, hastech=False):

    techniques = list(results.keys())
    models = list(results[techniques[0]].keys()) if hastech else list(results.keys())


    if hastech:
        accuracies = {model: [results[technique][model]['accuracy'] for technique in techniques]
                      for model in models}
    else:
        accuracies = {model: [results[model]['accuracy']] for model in models}

    # Plot
    plt.figure(figsize=(12, 6))
    if hastech:
        x = np.arange(len(techniques))
        width = 0.25
        for i, (model, acc) in enumerate(accuracies.items()):
            plt.bar(x + i * width, acc, width, label=model)
        plt.xticks(x + width, techniques)
    else:
        x = np.arange(len(models))
        plt.bar(x, [acc[0] for acc in accuracies.values()])
        plt.xticks(x, models)

    plt.xlabel('Balancing Techniques' if hastech else 'Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()

results_no_pca, hastech_no_pca = train_evaluate_supervised_models(X, Y, with_pca=False)
results_with_pca, hastech_with_pca = train_evaluate_supervised_models(X, Y, with_pca=True)
results_balancing, hastech_balancing = evaluate_models_with_balancing(X, Y)
results_pca_balancing, hastech_pca_balancing = evaluate_models_with_pca_and_balancing(X, Y)

plot_accuracy_comparison(results_no_pca, hastech_no_pca)
plot_accuracy_comparison(results_with_pca, hastech_with_pca)
plot_accuracy_comparison(results_balancing, hastech_balancing)
plot_accuracy_comparison(results_pca_balancing, hastech_pca_balancing)


def find_optimal_clusters(X, max_k=10):
    visualizer = KElbowVisualizer(KMeans(random_state=42), k=(1, max_k))
    visualizer.fit(X)
    plt.title('Elbow Method for Optimal k')
    plt.show()
    return visualizer.elbow_value_

# Find optimal number of clusters
optimal_k = find_optimal_clusters(X)
print(f"\nOptimal number of clusters: {optimal_k}")

# K-Means Clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X)

# Evaluate clustering
silhouette_avg = silhouette_score(X, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg}")

'''
Unbalanced Technique:
Logistic Regression: 83.62% (0.8361581920903954)
SMOTE Technique:
Logistic Regression: 83.33% (0.8333333333333334)
Random Over-sampling Technique:
Random Forest: 83.85% (0.8385416666666666)
Random Under-sampling Technique:
SVM: 82.72% (0.8271604938271605)
Unbalanced Technique with PCA:
Logistic Regression: 80.79% (0.807909604519774)
SMOTE Technique with PCA:
Logistic Regression: 83.85% (0.8385416666666666)
Random Over-sampling Technique with PCA:
Random Forest: 83.85% (0.8385416666666666)
Random Under-sampling Technique with PCA:
Random Forest: 80.86% (0.808641975308642)

Highest With 83.85% (0.8385416666666666):
SMOTE Technique with PCA (Logistic Regression)
Random Over-sampling Technique (Random Forest)
Random Over-sampling Technique with PCA (Random Forest)

'''