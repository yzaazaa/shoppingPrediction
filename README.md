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
python shopping.py shopping.csv
```
**Example**
```python
 if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")
```

**Output Example:**
```bash
Correct: 4067
Incorrect: 865
True Positive Rate: 38.65%
True Negative Rate: 90.36%
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
