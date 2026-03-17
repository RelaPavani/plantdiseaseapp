from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime

app = Flask(__name__)

# ✅ DATABASE CONFIGURATION
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///predictions.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ✅ DATABASE TABLE
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    temperature = db.Column(db.Float)
    humidity = db.Column(db.Float)
    moisture = db.Column(db.Float)
    ph = db.Column(db.Float)
    light = db.Column(db.Float)
    result = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.now)

# ✅ LOAD DATASET & TRAIN MODEL
data = pd.read_csv("plant_disease_dataset.csv")
X = data.drop("Disease", axis=1)
y = data["Disease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# ✅ HOME PAGE
@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = ""
    confidence_val = None

    if request.method == "POST":
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        moisture = float(request.form["moisture"])
        ph = float(request.form["ph"])
        light = float(request.form["light"])

        sample = pd.DataFrame(
            [[temperature, humidity, moisture, ph, light]],
            columns=["Temperature", "Humidity", "Moisture", "pH", "LightHours"]
        )

        prediction = model.predict(sample)
        probability = model.predict_proba(sample)
        confidence_val = round(max(probability[0]) * 100, 2)

        prediction_text = "Diseased Plant ⚠️" if prediction[0] == 1 else "Healthy Plant ✅"

        # Save to database
        new_entry = Prediction(
            temperature=temperature,
            humidity=humidity,
            moisture=moisture,
            ph=ph,
            light=light,
            result=prediction_text,
            confidence=confidence_val
        )
        db.session.add(new_entry)
        db.session.commit()

    return render_template("index.html", prediction=prediction_text, confidence=confidence_val)

# ✅ HISTORY PAGE
@app.route("/history")
def history():
    records = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    healthy_count = Prediction.query.filter_by(result="Healthy Plant ✅").count()
    diseased_count = Prediction.query.filter_by(result="Diseased Plant ⚠️").count()
    return render_template("history.html", records=records, healthy_count=healthy_count, diseased_count=diseased_count)

# ✅ EXCEL UPLOAD PAGE
@app.route("/upload", methods=["GET", "POST"])
def upload():
    total = healthy = diseased = healthy_percent = diseased_percent = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            import io
            df = pd.read_excel(io.BytesIO(file.read()), engine="openpyxl")
            df.columns = df.columns.str.strip()

            required_cols = ["Temperature", "Humidity", "Moisture", "pH", "LightHours"]
            if not all(col in df.columns for col in required_cols):
                return "Excel columns must be: Temperature, Humidity, Moisture, pH, LightHours"

            df = df[required_cols]
            predictions = model.predict(df)
            probabilities = model.predict_proba(df)

            healthy = diseased = 0

            for i in range(len(df)):
                confidence = round(max(probabilities[i]) * 100, 2)
                result_text = "Diseased Plant ⚠️" if predictions[i] == 1 else "Healthy Plant ✅"

                if predictions[i] == 1:
                    diseased += 1
                else:
                    healthy += 1

                record = Prediction(
                    temperature=float(df.iloc[i]["Temperature"]),
                    humidity=float(df.iloc[i]["Humidity"]),
                    moisture=float(df.iloc[i]["Moisture"]),
                    ph=float(df.iloc[i]["pH"]),
                    light=float(df.iloc[i]["LightHours"]),
                    result=result_text,
                    confidence=confidence
                )
                db.session.add(record)

            db.session.commit()

            total = healthy + diseased
            healthy_percent = round((healthy / total) * 100, 2)
            diseased_percent = round((diseased / total) * 100, 2)

    return render_template(
        "upload.html",
        total=total,
        healthy=healthy,
        diseased=diseased,
        healthy_percent=healthy_percent,
        diseased_percent=diseased_percent
    )

# ✅ MUST BE LAST
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)