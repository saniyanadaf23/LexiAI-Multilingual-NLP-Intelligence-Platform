# 🧠 **LexiAI — Multilingual NLP Intelligence Platform**

**LexiAI** is an advanced Natural Language Processing (NLP) web application that empowers users to analyze, understand, and process multilingual text using AI-driven linguistic techniques.  
It integrates **Flask (Backend)**, **Streamlit (Frontend)**, and **SQLite (Database)** to create a seamless and intelligent NLP ecosystem.

---

## 🚀 **Core Features**

### 👤 **User Authentication**
- Secure **Sign Up** and **Login** system with password hashing using Werkzeug.
- User credentials and session data managed by Flask.
- Automatically redirects to the Streamlit dashboard after successful login.

### 🧠 **NLP Functionalities**
- **Part of Speech Tagging (POS):** Linguistic structure analysis using Stanza.
- **Grammar Correction:** Detects and corrects English grammar errors.
- **Sentiment Analysis:** Identifies positive, negative, or neutral tone using TextBlob.
- **Named Entity Recognition (NER):** Detects people, organizations, and locations.
- **Word Cloud & Frequency:** Generates frequency-based visual representations.
- **Text Summarization:** Condenses large paragraphs into meaningful summaries.

### 💾 **User Analysis History**
- Each analysis result is automatically saved in the Flask database.
- Users can view their complete analysis history via the Streamlit sidebar.
- Every entry includes the analyzed text, NLP feature, and result.
- All user data is isolated based on their email.

### 🎨 **Modern Interface & UI Enhancements**
- **Glassmorphism UI:** Elegant translucent panels and glowing accents.
- **Unified Design System:** Consistent colors and typography across login, signup, and dashboard.
- **Animated Sidebar:** Interactive icons, glowing separators, and structured sections.
- **Flask Status Indicator:** Real-time 🟢 “Connected to Flask” status with soft pulsing glow.
- **Dark Theme:** Professional teal-on-dark color palette for a futuristic AI look.

---

## 🧩 **Tech Stack Overview**

| Component | Technology |
|------------|-------------|
| **Frontend (Login & Signup)** | HTML, CSS, JavaScript |
| **NLP Frontend Dashboard** | Streamlit |
| **Backend API & Auth** | Flask (Python) |
| **Database** | SQLite (SQLAlchemy ORM) |
| **NLP Libraries** | Stanza, TextBlob, SpaCy, Gensim |
| **Security** | Password Hashing (Werkzeug) |
| **CORS Handling** | Flask-CORS |

---

## ⚙️ **Installation & Setup Guide**

### 1️⃣ **Clone or Extract the Project**
```bash
cd your_project_folder

### Install dependencies
pip install flask flask-cors flask-sqlalchemy streamlit requests werkzeug

### Run the Flask Server
python flask_server.py

### You should see:
✅ Streamlit started on port 8501.
🌐 Flask running on http://localhost:5000
