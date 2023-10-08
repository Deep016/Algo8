import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
wine_data = pd.read_csv(url, sep=";")

# Check for missing values
missing_values = wine_data.isnull().sum()

# Convert quality to categorical
wine_data['quality'] = pd.Categorical(wine_data['quality'])

# Distribution of wine quality scores
plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=wine_data)
plt.title("Distribution of Wine Quality Scores")
plt.xlabel("Quality")
plt.ylabel("Count")
plt.show()

# Correlation heatmap
correlation_matrix = wine_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Explore the relationship between alcohol content and wine quality
plt.figure(figsize=(8, 5))
sns.boxplot(x='quality', y='alcohol', data=wine_data)
plt.title("Alcohol Content vs. Wine Quality")
plt.xlabel("Quality")
plt.ylabel("Alcohol Content")
plt.show()

# Split the data into features (X) and target (y)
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier to identify feature importance
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Calculate feature importances
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_classifier.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Generate classification report to assess model performance
y_pred = rf_classifier.predict(X_test)
report = classification_report(y_test, y_pred)

# Print feature importance and classification report
print("Feature Importance:")
print(feature_importances)
print("\nClassification Report:")
print(report)
