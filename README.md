# 🎯 Career Path Predictor using Machine Learning

## 📌 Project Overview
The **Career Path Predictor** is a machine learning-based web application that recommends suitable career roles based on a user's skills, interests, and academic preferences. The project uses multiple trained machine learning models and provides career predictions through an interactive **Streamlit** interface.

This project aims to help students and aspiring professionals explore career opportunities and make informed career decisions.

---

## 🚀 Features
- Predicts the most suitable career path based on user inputs.
- Interactive and user-friendly web interface built with Streamlit.
- Uses multiple machine learning algorithms for prediction.
- Attractive and responsive dashboard design.
- Fast and lightweight deployment.

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, LightGBM, CatBoost |
| Model Storage | Joblib |

---

## 🤖 Machine Learning Models Used
The application includes the following trained models:

1. Random Forest Classifier
2. XGBoost Classifier
3. LightGBM Classifier
4. CatBoost Classifier

The models were trained and saved using `.joblib` files for efficient loading during prediction.

---

## 📂 Project Structure

```text
ML-Project-main/
│── career_predictor_app.py          # Main Streamlit application
│── best_random_forest_model.joblib # Trained Random Forest model
│── best_xgboost_model.joblib       # Trained XGBoost model
│── lightgbm_model.joblib           # Trained LightGBM model
│── catboost_model.joblib           # Trained CatBoost model
│── requirements.txt               # Project dependencies
│── README.md                       # Project documentation
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository
```bash
git clone <repository-link>
cd ML-Project-main
```

### 2. Create a Virtual Environment (Optional)
```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**Linux/Mac**
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run career_predictor_app.py
```

---

## 📊 Workflow

1. User enters career-related information.
2. Input data is processed and prepared for prediction.
3. The trained machine learning model analyzes the data.
4. The application predicts the most suitable career path.
5. Results are displayed through the Streamlit dashboard.

---

## 🎯 Project Objectives
- Assist students in choosing suitable career paths.
- Apply machine learning techniques to career recommendation systems.
- Build an interactive and practical real-world application.
- Demonstrate the implementation of multiple classification models.

---

## 📈 Future Enhancements
- Add more career categories and datasets.
- Integrate personalized learning recommendations.
- Improve prediction accuracy with advanced ensemble techniques.
- Deploy the application on cloud platforms.

---

## 👨‍💻 Project Information
**Project Title:** Career Path Predictor using Machine Learning  
**Project Duration:** 16 June 2026 – 29 June 2026

---

## 📜 License
This project is developed for educational and academic purposes.
