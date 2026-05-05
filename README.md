# Stress Level Prediction Model

A machine learning project that predicts stress levels based on screen time, sleep, and exercise patterns. Built with a scikit-learn model, FastAPI backend and a modern frontend.

## 📋 Overview

This application combines a trained machine learning model with a FastAPI backend and interactive web frontend to predict an individual's stress level (Low, Medium, or High) based on their daily lifestyle habits. The model analyzes factors such as work screen time, leisure screen time, sleep duration, sleep quality, and exercise frequency.

## 🎯 Features

- **ML-powered Predictions**: Logistic Regression model trained on screen time and lifestyle data
- **REST API**: FastAPI server with CORS support for seamless integration
- **Interactive Web Interface**: User-friendly HTML/CSS/JS frontend for stress level predictions at ...
- **Model Optimization**: Uses SMOTE (Synthetic Minority Oversampling) to handle class imbalance
- **Hyperparameter Tuning**: GridSearchCV for optimal model performance
- **Health Check Endpoint**: Built-in endpoint to verify API availability

## 📂 Project Structure for both Backend with model and Frontend

```
backend+model/
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI application and endpoints
├── artifacts/
│   └── stress_level_prediction_model.joblib  # Trained model
├── train.py                 # Model training and evaluation script
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project configuration
├── ScreenTime_with_Stress_Classes.csv  # Training dataset
└── README.md
frontend/
├── stress_level.html       # Main web interface
├── stress_level.js         # Client-side logic
└── stress_prediction_new.css  # Styling
```

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn
- **ML/Data**: scikit-learn, pandas, numpy, imbalanced-learn
- **Frontend**: HTML5, CSS3, JavaScript
- **Model Serialization**: joblib
- **Visualization**: matplotlib, seaborn

## 📦 Installation

### Prerequisites

- Python 3.8+

### Steps

1. **Clone/Navigate to the project**

   ```bash
   cd backend+model
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv virt
   ```

3. **Activate the virtual environment**
   - **Windows (PowerShell)**:
     ```bash
     .\virt\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt)**:
     ```bash
     virt\Scripts\activate.bat
     ```
   - **Mac/Linux**:
     ```bash
     source virt/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Training the Model

Run the training script to train or retrain the stress level prediction model:

```bash
python train.py
```

This will:

- Load and preprocess the `ScreenTime_with_Stress_Classes.csv` dataset
- Apply SMOTE to handle class imbalance
- Train a Logistic Regression classifier
- Perform cross-validation and hyperparameter tuning
- Save the trained model to `artifacts/stress_level_prediction_model.joblib`

**Note**: The model must be trained at least once before running the API.

### Running the Backend API

Start the FastAPI development server:

```bash
python -m uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

- **API Documentation**: `http://localhost:8000/docs` (Swagger UI)
- **Alternative Docs**: `http://localhost:8000/redoc` (ReDoc)

### Using the Frontend

1. Open `frontend/stress_level.html` in your web browser
2. Enter your lifestyle data:
   - Work screen hours per day
   - Leisure screen hours per day
   - Average sleep hours per night
   - Sleep quality (1-5 scale)
   - Exercise minutes per week
3. Click "Predict" to receive your stress level prediction

## 📡 API Endpoints

### Health Check

- **Endpoint**: `GET /health`
- **Description**: Verify API is running
- **Response**:
  ```json
  { "status": "ok" }
  ```

### Stress Level Prediction

- **Endpoint**: `POST /predict`
- **Description**: Predict stress level based on lifestyle data
- **Request Body**:
  ```json
  {
    "work_screen_hours": 8.5,
    "leisure_screen_hours": 3.0,
    "sleep_hours": 7.0,
    "sleep_quality_1_5": 4,
    "exercise_minutes_per_week": 150
  }
  ```
- **Response**:

  ```json
  {
    "Predicted Stress Level": 1
  }
  ```

  - `0` = Low stress
  - `1` = Medium stress
  - `2` = High stress

## 📊 Model Details

- **Algorithm**: Logistic Regression
- **Training Data**: ScreenTime with Stress Classes dataset
- **Features Used**:
  - `work_screen_hours`: Daily work-related screen time
  - `leisure_screen_hours`: Daily leisure screen time
  - `sleep_hours`: Average hours of sleep per night
  - `sleep_quality_1_5`: Subjective sleep quality (1-5 scale)
  - `exercise_minutes_per_week`: Weekly exercise duration
- **Target Classes**: Low, Medium, High (stress levels)
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Oversampling Technique)
- **Preprocessing**: StandardScaler normalization
- **Validation**: Stratified K-Fold cross-validation

## 📈 Features Explanation

| Feature                   | Range   | Description                             |
| ------------------------- | ------- | --------------------------------------- |
| work_screen_hours         | 0-24    | Hours spent on work-related screen time |
| leisure_screen_hours      | 0-24    | Hours spent on leisure screen time      |
| sleep_hours               | 0-24    | Average hours of sleep per night        |
| sleep_quality_1_5         | 1-5     | Subjective sleep quality rating         |
| exercise_minutes_per_week | 0-10080 | Total weekly exercise duration          |

## 🔒 CORS Configuration

The API is configured with CORS support to allow requests from:

- All origins (`*`)
- All HTTP methods
- All headers
- Credential-based requests

For production, update `CORSMiddleware` in `app/main.py` to restrict origins as needed.

## 📝 Dataset

The model is trained on `ScreenTime_with_Stress_Classes.csv` which was obtained from Kaggle containing lifestyle and screen time data with corresponding stress level classifications. The dataset excludes certain features like user demographics to focus on behavioral factors.

## 🔗 Running Both Backend and Frontend Together locally

1. **Terminal 1 - Start the API**:

   ```bash
   python -m uvicorn app.main:app --reload
   ```

2. **Terminal 2 - Open the frontend**:
   - Open `../frontend/stress_level.html` in your browser
   - The frontend will automatically communicate with the backend at `http://localhost:8000`

## 👤 Authors

Munachimso Enabulele, Akinola Fortune and Ajagbe Oluwateniolafunmi
