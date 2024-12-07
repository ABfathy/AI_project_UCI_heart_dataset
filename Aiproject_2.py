import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, RandomOverSampler

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


