## 🎮 Movie Recommendation System

This is a Flask-based movie recommendation system that provides movie recommendations based on similarity models.

### 🚀 Features
- Flask API for serving recommendations
- Pre-trained models for similarity-based recommendations
- Dockerized for easy deployment
- Uses Gunicorn for production-ready performance

---

## 📦 Setup & Run Locally

### 1⃣ Install Dependencies
Ensure you have Python **3.10+** installed.

```bash
pip install -r requirements.txt
```

### 2⃣ Run Flask App Locally
```bash
python app.py
```

The app will be available at:  
👉 `http://127.0.0.1:5000/`

---

## 🐳 Running with Docker

### 1⃣ Build Docker Image
```bash
docker build -t flask-app .
```

### 2⃣ Run Container
```bash
docker run -p 5000:5000 flask-app
```

The app will be available at:  
👉 `http://localhost:5000/`

---

## ⚡ Deployment (Render or Other Platforms)
By default, the **Dockerfile** is optimized for deployment using **Gunicorn**. Modify `CMD` in the `Dockerfile` if needed.

---

