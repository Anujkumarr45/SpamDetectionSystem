from flask import Flask, render_template, request, redirect, url_for, session
from pymongo import MongoClient
from dotenv import load_dotenv
import pickle
import os

app = Flask(__name__)

# =========================
# Load Environment Variables
# =========================
load_dotenv()

# Secret key for session
app.secret_key = os.getenv("SECRET_KEY", "your_secret_key")


# =========================
# MongoDB Atlas Connection
# =========================
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI is not set in environment variables!")

client = MongoClient(MONGO_URI)

db = client["spam_detector_db"]

users = db["users"]

# Collection for storing SMS predictions
predictions = db["predictions"]


# =========================
# Load ML Model
# =========================
with open("model/spam_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("model/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# =========================
# Home
# =========================
@app.route("/")
def home():
    return redirect(url_for("login"))


# =========================
# Signup
# =========================
@app.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            return "Username and password are required!"

        # Check if username already exists
        if users.find_one({"username": username}):
            return "Username already exists!"

        # Save user
        users.insert_one({
            "username": username,
            "password": password
        })

        return redirect(url_for("login"))

    return render_template("signup.html")


# =========================
# Login
# =========================
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = users.find_one({
            "username": username,
            "password": password
        })

        if user:
            session["user"] = username

            return redirect(url_for("dashboard"))

        return "Invalid username or password!"

    return render_template("login.html")


# =========================
# Dashboard
# =========================
@app.route("/dashboard")
def dashboard():

    # User must be logged in
    if "user" not in session:
        return redirect(url_for("login"))

    username = session["user"]

    # Total messages checked by current user
    total_messages = predictions.count_documents({
        "username": username
    })

    # Total spam messages
    spam_messages = predictions.count_documents({
        "username": username,
        "prediction": "Spam"
    })

    # Total ham messages
    ham_messages = predictions.count_documents({
        "username": username,
        "prediction": "Ham"
    })

    # Spam percentage
    if total_messages > 0:
        spam_percentage = round(
            (spam_messages / total_messages) * 100,
            2
        )
    else:
        spam_percentage = 0

    # Load model accuracy
    with open("model/accuracy.pkl", "rb") as file:
        accuracy = pickle.load(file)

    model_accuracy = round(accuracy * 100, 2)

    return render_template(
        "dashboard.html",
        username=username,
        total_messages=total_messages,
        spam_messages=spam_messages,
        ham_messages=ham_messages,
        spam_percentage=spam_percentage,
        model_accuracy=model_accuracy
    )


# =========================
# SMS Spam Detector
# =========================
@app.route("/sms", methods=["GET", "POST"])
def sms_detector():

    # User must be logged in
    if "user" not in session:
        return redirect(url_for("login"))

    prediction = None
    confidence = None

    if request.method == "POST":

        # Get message from form
        message = request.form.get("message", "").strip()

        if not message:
            return render_template(
                "index.html",
                prediction=None,
                confidence=None,
                error="Please enter a message!"
            )

        # Convert message into TF-IDF vector
        data = vectorizer.transform([message])

        # Predict
        result = model.predict(data)[0]

        # Convert 0/1 into readable result
        if result == 1:
            prediction = "Spam"
        else:
            prediction = "Ham"

        # Calculate confidence
        probabilities = model.predict_proba(data)[0]
        confidence = max(probabilities) * 100

        # Save prediction to MongoDB
        predictions.insert_one({
            "username": session["user"],
            "message": message,
            "prediction": prediction,
            "confidence": round(confidence, 2)
        })

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(confidence, 2)
        if confidence is not None else None
    )


# =========================
# Logout
# =========================
@app.route("/logout")
def logout():

    session.pop("user", None)

    return redirect(url_for("login"))


# =========================
# Run Flask App
# =========================
if __name__ == "__main__":
    app.run(debug=True)