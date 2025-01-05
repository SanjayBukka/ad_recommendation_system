# README: Random Forest Model with Recommendation System

## Overview
This project involves developing a **Random Forest model** to predict the likelihood of a user clicking on an advertisement. We further integrated a **Recommendation System** that uses the model’s predictions to suggest relevant ads to users. Below is a detailed explanation of the steps taken from the beginning of the project to the current stage.

---

## Goals of the Project
1. Build a machine learning model to predict whether a user will click on an advertisement.
2. Use the predictions from the model to develop a recommendation system for targeted ads.
3. Ensure the system handles both training and real-time predictions effectively.

---

## Key Features
- **Random Forest Classifier**: Used for binary classification (click or no click).
- **One-Hot Encoding**: Handled categorical data to make it suitable for the model.
- **Data Handling**: Processed user behavior data, including attributes such as age, gender, browsing history, device type, and time of day.
- **Recommendation System**: Suggested ads to users based on model predictions.
- **Evaluation Metrics**: Included accuracy, confusion matrix, and classification report to assess the model.

---

## Steps Completed

### Step 1: Dataset Preparation
- **Dataset Description**:
  - Columns: `age`, `gender`, `device_type`, `ad_position`, `browsing_history`, `time_of_day`, `click` (target).
  - Target Variable: `click` (1 for clicked, 0 for not clicked).
- Preprocessed the dataset to:
  - Handle missing values.
  - Encode categorical features using **One-Hot Encoding**.
  - Split the data into training and test sets (80% training, 20% testing).

### Step 2: Model Selection and Training
- Used the **Random Forest Classifier** due to its robustness and ability to handle categorical and numerical features.
- Hyperparameters:
  - Number of estimators: 100.
  - Random state: 42.
- Trained the model on the preprocessed training data.

### Step 3: Model Evaluation
- Evaluated the model on the test data using:
  - **Accuracy Score**: Achieved 65.10%.
  - **Confusion Matrix**: Provided insights into false positives and false negatives.
  - **Classification Report**: Displayed precision, recall, and F1 scores for both classes (click, no click).
- Results showed:
  - Precision and recall were higher for the positive class (users likely to click).
  - Opportunities for improvement in handling imbalanced data.

### Step 4: Handling Imbalanced Data
- Addressed class imbalance by **resampling the training data**:
  - Used oversampling to balance positive and negative classes.
- Retrained the model with resampled data, improving prediction performance on the minority class.

### Step 5: Developing the Recommendation System
- Designed a **real-time recommendation pipeline**:
  - Collected user data (e.g., age, gender, browsing history).
  - Processed the data to ensure compatibility with the trained model.
  - Used the trained model to predict the likelihood of a click.
  - Recommended ads to users with a high probability of clicking.
- Implemented safeguards:
  - Ensured the test-time input data matched the feature set used during training.

### Step 6: Real-Time User Predictions
- Created an interactive system where:
  - A new user’s data is processed and encoded in real-time.
  - The model predicts the likelihood of a click.
  - If the prediction is positive, an ad is recommended.
- Example User Data:
  ```python
  user_data = pd.DataFrame({
      'age': [25],
      'gender': ['Female'],
      'device_type': ['Mobile'],
      'ad_position': ['Top'],
      'browsing_history': ['Shopping'],
      'time_of_day': ['Morning']
  })
  ```
- Encoded the user data to match the training features and used the model for predictions.

### Step 7: Recommendations Logic
- Integrated a rule-based recommendation logic:
  - If the predicted likelihood is high, recommend relevant ads.
  - Personalized recommendations based on user attributes (e.g., age, gender, browsing history).
- Future Enhancements:
  - Add **Collaborative Filtering**: Recommend ads based on user similarity.
  - Add **Content-Based Filtering**: Suggest ads based on item characteristics.

---

## Tools and Libraries Used
1. **Python**: Programming language for data processing and modeling.
2. **Pandas**: For data manipulation and preprocessing.
3. **Scikit-learn**: For model building, training, and evaluation.
4. **NumPy**: For numerical computations.

---

## Results
- **Model Accuracy**: 65.10%.
- **Confusion Matrix**:
  ```
  [[425 280]
   [418 877]]
  ```
- **Classification Report**:
  - Precision: 0.76 for class 1 (click).
  - Recall: 0.68 for class 1 (click).
  - Weighted F1 Score: 0.66.

---

## Future Work
1. **Enhancing the Recommendation System**:
   - Add Collaborative Filtering and Content-Based Filtering for more personalized recommendations.
2. **Improving Model Performance**:
   - Experiment with other algorithms like Gradient Boosting or XGBoost.
   - Perform hyperparameter tuning to optimize the Random Forest model.
3. **Integrating the System**:
   - Build a user-friendly interface to collect real-time user data.
   - Deploy the system as a web application for real-world testing.

---

## Conclusion
This project successfully implemented a Random Forest model to predict ad clicks and laid the foundation for a recommendation system. The system provides personalized ad recommendations based on user data and model predictions. With further enhancements, it can be deployed as a robust tool for targeted advertising.


