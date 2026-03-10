from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
# ✅ LOAD DATASET & TRAIN MODEL
data = pd.read_csv("plant_disease_dataset.csv")
X = data.drop("Disease", axis=1)
y = data["Disease"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
# ✅ HOME PAGE
@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = ""

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
        confidence = round(max(probability[0]) * 100, 2)
        print("Confidence:", confidence)

        if prediction[0] == 1:
            prediction_text = "Diseased Plant ⚠️"
        else:
            prediction_text = "Healthy Plant ✅"

        # ✅ Save to database
        new_entry = Prediction(
            temperature=temperature,
            humidity=humidity,
            moisture=moisture,
            ph=ph,
            light=light,
            result=prediction_text
        )

        db.session.add(new_entry)
        db.session.commit()

    return render_template("index.html", prediction=prediction_text)

# ✅ HISTORY PAGE
@app.route("/history")
def history():
    records = Prediction.query.all()
    return render_template("history.html", records=records)

# ✅ EXCEL UPLOAD PAGE
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]

        if file:
            df = pd.read_excel(file, engine="openpyxl")
            df.columns = df.columns.str.strip()

            required_cols = ["Temperature", "Humidity", "Moisture", "pH", "LightHours"]
            df = df[required_cols]

            predictions = model.predict(df)
            for i in range(len(df)):
                result_text = "Diseased Plant ⚠️" if predictions[i] == 1 else "Healthy Plant ✅"

                record = Prediction(
                    temperature=float(df.iloc[i]["Temperature"]),
                    humidity=float(df.iloc[i]["Humidity"]),
                    moisture=float(df.iloc[i]["Moisture"]),
                    ph=float(df.iloc[i]["pH"]),
                    light=float(df.iloc[i]["LightHours"]),
                    result=result_text
                )

                db.session.add(record)

            db.session.commit()

            total = len(predictions)
            diseased = sum(predictions)
            healthy = total - diseased

            total = int(total)
            healthy = int(healthy)
            diseased = int(diseased)

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

    return render_template("upload.html")

# ✅ MUST BE LAST
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
