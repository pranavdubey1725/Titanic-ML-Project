# Titanic Survival Prediction

## OVERVIEW
- Predict passenger survival on the Titanic disaster using historical passenger data.
- Involves data cleaning, preprocessing, and strategic feature engineering.
- Applies machine learning classification models: **Logistic Regression** and **K-Nearest Neighbors (KNN)**.
- Evaluates model performance using accuracy metrics and validates through Kaggle competition submission.

---

## PROJECT STRUCTURE
- **TitanicSurvivalPrediction.ipynb** – Main notebook for data preprocessing, feature engineering, model training, and evaluation.
- **train.csv** – Dataset containing historical Titanic passenger records.
- **test.csv** – Test dataset for prediction and Kaggle submission.
- **logistic_regression_model.pkl** – Saved Logistic Regression trained model.
- **knn_model.pkl** – Saved KNN trained model.
- **requirements.txt** – List of project's Python dependencies.
- **README.md** – Project documentation.

---

## DATASET USED
- **train.csv** — Training dataset containing passenger records with features related to demographics, ticket class, and family information.
- **test.csv** — Test dataset for generating predictions and Kaggle competition submission.
- Engineered features include **FamilySize** (combining SibSp and Parch), **IsAlone** (binary indicator), and **Title** (extracted from passenger names).
- Categorical variables were encoded and missing values were imputed for model compatibility.

---

## FEATURES USED
- **Passenger Class (Pclass)** – Socio-economic status indicator
- **Age** – Passenger age with missing values imputed
- **Sex** – Gender of the passenger
- **Family Size** – Total number of family members aboard
- **IsAlone** – Binary feature indicating if passenger traveled alone
- **Title** – Extracted from passenger name (Mr., Mrs., Miss., Master, etc.)

---

## TECH STACK
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Techniques:** Feature Engineering, Missing Value Imputation, Classification Modeling, Cross-Validation, Model Evaluation
- **Environment:** Jupyter Notebook / Google Colab
- **Platform:** Kaggle Competition

---

## MODEL PERFORMANCE
Final model scores were evaluated using accuracy metrics on local test data and validated through Kaggle public leaderboard.

| Model | Local Test Accuracy | Kaggle Public Score |
| :--- | :--- | :--- |
| Logistic Regression | ~79% | 0.78947 |
| K-Nearest Neighbors | ~79% | 0.78947 |

---

## WORKFLOW
1. **Data Cleaning and Preprocessing** – Handle missing values and prepare data for analysis  
2. **Exploratory Data Analysis (EDA)** – Visualize patterns and relationships in the data  
3. **Feature Engineering** – Create new features to improve model performance  
4. **Model Training and Evaluation** – Train classification models and assess accuracy  
5. **Kaggle Submission** – Generate predictions on test data and submit to competition  

---

## KEY INSIGHTS
- Gender was the strongest predictor of survival (women had higher survival rates)
- Passenger class significantly impacted survival probability
- Passengers traveling alone had different survival patterns compared to those with family
- Age and title extraction provided additional predictive power

---

## FUTURE IMPROVEMENTS
- Experiment with ensemble methods (Random Forest, XGBoost, Gradient Boosting)
- Perform advanced hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Explore additional feature interactions and polynomial features
- Implement stacking and blending techniques for improved accuracy
