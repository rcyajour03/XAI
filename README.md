XAI Assistant ML
This project is an implementation of an Explainable AI (XAI) system for machine learning tasks, providing insights into model predictions. Follow the instructions below to set up and run the application.

Setup and Run Instructions

1. Activate the Virtual Environment
Navigate to the project directory using the terminal or command prompt.
Activate the virtual environment using the following command:
bash
Copy code
.\XAI_Assistant_ML\Scripts\Activate
2. Run the Code
After activating the virtual environment, run the Streamlit application with:
bash
Copy code
streamlit run main.py
Application Interface Screenshots
Here are a few screenshots of the application interface:

Homepage:![image](https://github.com/user-attachments/assets/1766facc-812e-4712-ab4a-debfd78c246e)


Sidebar:![image](https://github.com/user-attachments/assets/6a9c571b-3c82-49db-899c-c7bcf72296ba)


Results Panel:

![Screenshot 2024-11-18 151146](https://github.com/user-attachments/assets/1746afb0-1fb4-45e9-ac4a-a0057de15878)

![image](https://github.com/user-attachments/assets/1594953e-654b-43d5-b3e5-9e6d25ddf19f)


Requirements
Ensure you have the following installed:

Python 3.8+
Streamlit
You can install the required dependencies by running:

bash
Copy code
pip install -r requirements.txt


## Architecture

The Explainable AI-Based Disease Prediction System is designed to accurately predict
diseases while ensuring the interpretability of its decisions. The process begins with data
pre-processing, where raw data is cleaned by handling missing values, removing outliers,
and normalizing the features. The dataset is then divided into a training set (80%)
and a testing set (20%) for model evaluation. To optimize performance, the feature
selection technique Elastic Net which combines Lasso (L1 regularization), and Ridge (L2
regularization) is applied to identify the most relevant features. The system incorporates
n-fold cross-validation to validate the model using different splits of the dataset, ensuring
robust evaluation. The machine learning models Deep Neural Networks (DNN), Random
Forest, Support Vector Machines (SVM), and K-Nearest Neighbors (KNN) are tested and
hyperparameters are fine-tuned to maximize accuracy. For interpretability, the system
employs five Explainable AI (XAI) techniques viz., Decision Trees, SHAP (SHapley
Additive Explanations), LIME (Local Interpretable Model-Agnostic Explanations), and
the Explainable Boosting Algorithm to provide clear insights into model predictions. The
system allows users to input their data, generates a disease prediction, and offers an
explanation for the outcome, ensuring trust and transparency. The output is a front-end
dashboard that is developed for healthcare professionals and general public to interact
with the system, view predictions, and understand the model’s reasoning through visual
and textual explanations.

![Database](https://github.com/user-attachments/assets/bc4f9a55-552b-4402-8bc4-5735404b8843)


## Model Performance Results

### Alzheimer's Disease Model

| Model                | Recall   | Accuracy  | Precision   | F1 Score   |
|----------------------|----------|-----------|-------------|------------|
| EBC                  | 0.75     | 0.92      | 0.9501      | 0.7738     |
| Classification Tree  | 0.704    | 0.867     | 0.7262      | 0.7110     |
| Lime                 | 0.741    | 0.907     | 0.9388      | 0.7639     |
| Shap                 | 0.741    | 0.907     | 0.9388      | 0.7639     |
| Logistic Regression  | 0.75     | 0.92      | 0.9501      | 0.7738     |

---

### Heart Disease Model

| Model                | Recall   | Accuracy  | Precision   | F1 Score   |
|----------------------|----------|-----------|-------------|------------|
| EBC                  | 0.96     | 0.85      | 0.8         | 0.8767     |
| Classification Tree  | 0.818    | 0.754     | 0.75        | 0.7826     |
| Lime                 | 0.939    | 0.8196    | 0.775       | 0.8493     |
| Shap                 | 0.939    | 0.9197    | 0.775       | 0.8493     |
| Logistic Regression  | 0.803    | 0.909     | 0.7692      | 0.8333     |

---

### Diabetes Disease Model

| Model                | Recall   | Accuracy  | Precision   | F1 Score   |
|----------------------|----------|-----------|-------------|------------|
| EBC                  | 0.6296   | 0.7597    | 0.6666      | 0.6476     |
| Classification Tree  | 0.2593   | 0.6948    | 0.666       | 0.3733     |
| Lime                 | 0.5741   | 0.7273    | 0.62        | 0.5962     |
| Shap                 | 0.5741   | 0.7273    | 0.62        | 0.5962     |
| Logistic Regression  | 0.5185   | 0.7143    | 0.6087      | 0.56       |

