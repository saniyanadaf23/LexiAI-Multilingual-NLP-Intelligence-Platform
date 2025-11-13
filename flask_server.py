from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import subprocess, time, os

# --- Flask setup ---
app = Flask(__name__, static_folder='signup-static', template_folder='signup-static')
CORS(app)

# --- Database setup ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- DB Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True)
    password_hash = db.Column(db.String(200))

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120))
    text = db.Column(db.Text)
    feature = db.Column(db.String(50))
    result = db.Column(db.Text)

# Create tables if not exist
with app.app_context():
    db.create_all()

# --- Routes ---
@app.route('/')
def index():
    return send_from_directory('signup-static', 'index.html')

@app.route('/signup', methods=['POST'])
def signup():
    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")

    if not all([name, email, password]):
        return jsonify({"status": "error", "message": "All fields required."}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"status": "error", "message": "Email already exists."}), 400

    hashed_pw = generate_password_hash(password)
    user = User(name=name, email=email, password_hash=hashed_pw)
    db.session.add(user)
    db.session.commit()

    return jsonify({"status": "success", "message": "Account created successfully!"})

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get("email")
    password = request.form.get("password")

    user = User.query.filter_by(email=email).first()
    if user and check_password_hash(user.password_hash, password):
        return jsonify({
            "status": "success",
            "redirect": f"http://localhost:8501/?email={email}",
            "email": email
        })
    return jsonify({"status": "error", "message": "Invalid credentials."}), 401

# API to save analysis results
@app.route('/save_analysis', methods=['POST'])
def save_analysis():
    data = request.get_json()
    entry = Analysis(
        user_email=data.get("email"),
        text=data.get("text"),
        feature=data.get("feature"),
        result=data.get("result")
    )
    db.session.add(entry)
    db.session.commit()
    return jsonify({"status": "success"})

# API to fetch analysis history
@app.route('/get_history/<email>', methods=['GET'])
def get_history(email):
    records = Analysis.query.filter_by(user_email=email).all()
    return jsonify([{
        "text": r.text,
        "feature": r.feature,
        "result": r.result
    } for r in records])

# --- Serve static files ---
@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('signup-static', filename)

# --- Launch Streamlit ---
if __name__ == '__main__':
    time.sleep(2)
    try:
        subprocess.Popen(["streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8501"])
        print("✅ Streamlit started on port 8501.")
    except Exception as e:
        print("⚠️ Could not start Streamlit:", e)

    print("🌐 Flask running on http://localhost:5000")
    app.run(port=5000, debug=True)
