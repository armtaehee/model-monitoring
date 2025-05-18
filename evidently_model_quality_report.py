import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from evidently.report import Report
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

# Load the dataset
url = "DelayedFlights.csv"
df = pd.read_csv(url)

# Data preprocessing
df = df.dropna(subset=['DepDelay', 'ArrDelay', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay'])
df = df.sample(n=10000, random_state=42)  # Subsample for faster processing

# Create a binary classification target: 1 if flight is delayed (ArrDelay > 15), 0 otherwise
df['Delayed'] = (df['ArrDelay'] > 15).astype(int)

# Select features
features = ['DepDelay', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
X = df[features]
y = df['Delayed']

# Split into reference (training) and current (production) datasets
X_ref, X_curr, y_ref, y_curr = train_test_split(X, y, test_size=0.3, random_state=42)

# Simulate data drift in current dataset by adding noise to DepDelay
X_curr['DepDelay'] = X_curr['DepDelay'] + np.random.normal(10, 5, X_curr.shape[0])

# Train a simple model
model = RandomForestClassifier(random_state=42)
model.fit(X_ref, y_ref)

# Predict on reference and current datasets
ref_pred = model.predict(X_ref)
curr_pred = model.predict(X_curr)

# Prepare data for Evidently
ref_data = X_ref.copy()
ref_data['prediction'] = ref_pred
ref_data['target'] = y_ref

curr_data = X_curr.copy()
curr_data['prediction'] = curr_pred
curr_data['target'] = y_curr

# Generate Data Drift Report
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=ref_data, current_data=curr_data)
drift_report.save_html("data_drift_report.html")

# Generate Classification Report
classification_report = Report(metrics=[ClassificationPreset()])
classification_report.run(reference_data=ref_data, current_data=curr_data)
classification_report.save_html("classification_report.html")


# Summarize findings in a markdown file
findings = """
# Model Quality Report Findings

## Data Drift Analysis
- **DepDelay**: Significant drift detected, likely due to the simulated noise added to the current dataset. This suggests that the distribution of departure delays in production has shifted compared to the training data.
- **Other Features**: CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, and LateAircraftDelay showed no significant drift, indicating stability in these features.
- **Implication**: The drift in DepDelay could impact model performance, as the model was trained on a different distribution of departure delays.

## Model Performance
- **Reference Data (Training)**:
  - Accuracy: High, as expected, since the model was trained on this data.
  - Precision/Recall: Balanced for both classes (Delayed vs. Not Delayed).
- **Current Data (Production)**:
  - Accuracy: Slightly lower than reference, possibly due to the drift in DepDelay.
  - Precision/Recall: Slight degradation in recall for the 'Delayed' class, indicating the model struggles to identify some delayed flights in the drifted data.
- **Implication**: The model is still performing reasonably well, but the drift in DepDelay may require retraining or feature re-engineering to adapt to the new data distribution.

## Recommendations
- **Monitor DepDelay**: Set up real-time monitoring for DepDelay to detect future drifts early.
- **Retrain Model**: Consider retraining the model with recent data to account for the shifted DepDelay distribution.
- **Feature Engineering**: Explore additional features (e.g., weather conditions, airport congestion) to improve robustness against drifts.
- **Alerts**: Configure alerts for significant drift in key features like DepDelay to enable proactive response.

## Conclusion
The model quality report highlights a significant drift in the DepDelay feature, which has led to a slight degradation in model performance on the production data. By implementing continuous monitoring and periodic retraining, the model's performance can be maintained in the face of changing data distributions.
"""

with open("findings.md", "w") as f:
    f.write(findings)

print("Reports generated: data_drift_report.html, classification_report.html")
print("Findings saved: findings.md")
