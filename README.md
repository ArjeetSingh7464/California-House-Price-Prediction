# California House Price Prediction

A Machine Learning web application that predicts California house prices using user input features.

## Project Overview

This project uses a trained Machine Learning model built with Scikit-learn to predict house prices in California based on housing-related features.

The application includes:

* FastAPI backend
* Streamlit frontend
* Machine Learning prediction system
* REST API integration

---

# Features

* Predict California house prices
* User-friendly web interface
* FastAPI backend API
* Streamlit frontend
* Real-time prediction
* Production-ready structure

---

# Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* FastAPI
* Streamlit
* Joblib
* Uvicorn

---

# Project Structure

```bash
house-price-app/
│
├── backend/
│   ├── api.py
│   ├── model.pkl
│   ├── pipeline.pkl
│
├── frontend/
│   ├── app.py
│
├── requirements.txt
├── README.md
```

---

# Machine Learning Workflow

1. Data preprocessing
2. Feature engineering
3. Model training
4. Model serialization using Joblib
5. Backend API creation
6. Frontend integration
7. Deployment

---

# Installation

## Clone Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
cd house-price-app
```

---

# Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Backend (FastAPI)

```bash
uvicorn backend.api:app --reload
```

Backend will run on:

```bash
http://127.0.0.1:8000
```

API Documentation:

```bash
http://127.0.0.1:8000/docs
```

---

# Run Frontend (Streamlit)

```bash
streamlit run frontend/app.py
```

Frontend will run on:

```bash
http://localhost:8501
```

---

# Input Features

The model uses the following features:

* Median Income
* House Age
* Average Rooms
* Average Bedrooms
* Population
* Average Occupancy
* Latitude
* Longitude

---

# Deployment

## Backend Deployment

* Render

## Frontend Deployment

* Streamlit Cloud

---

# Future Improvements

* Add database support
* Add user authentication
* Improve UI design
* Add model monitoring
* Docker deployment
* Cloud deployment using AWS

---

# Author

ARJEET SINGH

---

# License

This project is for learning and educational purposes.
