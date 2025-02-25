# Shopping Prediction

**Introduction**

This Python implementation provides a machine learning model for predicting whether a customer will make a purchase based on various shopping-related features. Key features include:

* **Data Preprocessing:**
  * Load and clean shopping data from a CSV file.
  * Convert categorical data into numerical format for machine learning.

* **Model Training:**
  * Train a k-nearest neighbors (k-NN) classifier to predict shopping behavior.
  * Evaluate model performance using accuracy and other metrics.

* **Prediction:**
  * Use the trained model to predict whether a user is likely to make a purchase.

**Usage**
1. Clone the Repository:
```bash
git clone https://github.com/yzaazaa/shoppingPrediction
cd shoppingPrediction
```
2. Install Dependencies

```bash
pip install -r requirements.txt
```
4. Run the script:
```
python shopping.py
```
**Example**
```python
from shopping import load_data, train_model

# Load dataset
evidence, labels = load_data("shopping.csv")

# Train model
model = train_model(evidence, labels)

# Example prediction
prediction = model.predict([evidence[0]])
print("Prediction:", prediction)
```

**Output Example:**
```bash
Prediction: [1]
```
(A prediction of 1 indicates the customer is likely to make a purchase, while 0 indicates they are not.)

**API Reference**

The script provides functions for handling the shopping dataset and training the prediction model.

* **load_data(filename):**

* Loads and preprocesses shopping data from a CSV file.
* Returns structured evidence and corresponding labels.

* **train_model(evidence, labels):**

* Trains a k-NN classifier using the given dataset.
* Returns the trained model.

* **evaluate(labels, predictions):**

* Computes model accuracy based on true labels and predictions.