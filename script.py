import os
import json
from flask import Flask, jsonify
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# ===== NLTK setup =====
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('indonesian'))

def preprocess_text(text):
    if isinstance(text, str):
        words = text.lower().split()
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        return ' '.join(words)
    return ''

# ===== Flask setup =====
app = Flask(__name__)

# ===== Google Sheet Connection =====
# Opsi 1: pakai file JSON di repo
SERVICE_ACCOUNT_FILE = "mnrca-473305-96c2262a9b7d.json"

# Opsi 2: pakai environment variable (lebih aman)
# SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT")
# creds = Credentials.from_service_account_info(json.loads(SERVICE_ACCOUNT_JSON), scopes=["https://www.googleapis.com/auth/spreadsheets"])

creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=["https://www.googleapis.com/auth/spreadsheets"])
client = gspread.authorize(creds)

SPREADSHEET_KEY = '1DmA5tgK2d-zusPEvNda2sNkprNjabIUs-oyt4cn6QEg'
spreadsheet = client.open_by_key(SPREADSHEET_KEY)
sheet_model = spreadsheet.worksheet("Model")
sheet_data = spreadsheet.worksheet("Raw")
sheet_catlist = spreadsheet.worksheet("Category List")

# ===== Load Data =====
df_model = pd.DataFrame(sheet_model.get_all_records())
df_data = pd.DataFrame(sheet_data.get_all_records())
df_catlist = pd.DataFrame(sheet_catlist.get_all_records())

df_model.dropna(subset=["Detail Problem (SDG)"], inplace=True)
df_model["Detail Problem (SDG)"] = df_model["Detail Problem (SDG)"].apply(preprocess_text)

# ===== Train Models =====
def train_nb_model(column):
    df_temp = df_model.dropna(subset=[column])
    X = df_temp["Detail Problem (SDG)"]
    y = df_temp[column]
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    grid = GridSearchCV(model, {'multinomialnb__alpha': [0.1, 1.0, 10.0]}, cv=5)
    grid.fit(X, y)
    return grid.best_estimator_

model_category = train_nb_model("Category")
model_area = train_nb_model("Area")
model_main = train_nb_model("Main Category")

# ===== Flask Routes =====
@app.route("/")
def home():
    return "Server running!"

@app.route("/classify", methods=["GET"])
def classify_data():
    df_unclassified = df_data[df_data["Main Category"] == ""].copy()
    if len(df_unclassified) == 0:
        return jsonify({"message": "Semua data sudah diklasifikasi."})

    df_unclassified["Cleaned"] = df_unclassified["Detail Problem (SDG)"].apply(preprocess_text)

    # Predict Category
    df_unclassified["Pred_Category"] = model_category.predict(df_unclassified["Cleaned"])

    # Predict Area (with constraint)
    pred_area = []
    for text, cat in zip(df_unclassified["Cleaned"], df_unclassified["Pred_Category"]):
        valid_areas = df_catlist[df_catlist["Category"] == cat]["Area"].dropna().unique()
        proba = model_area.predict_proba([text])[0]
        classes = model_area.classes_
        filtered = {cls: p for cls, p in zip(classes, proba) if cls in valid_areas}
        pred_area.append(max(filtered, key=filtered.get) if filtered else classes[proba.argmax()])
    df_unclassified["Pred_Area"] = pred_area

    # Predict Main Category (with constraint)
    pred_main = []
    for text, cat in zip(df_unclassified["Cleaned"], df_unclassified["Pred_Category"]):
        valid_main = df_catlist[df_catlist["Category"] == cat]["Main Category"].dropna().unique()
        proba = model_main.predict_proba([text])[0]
        classes = model_main.classes_
        filtered = {cls: p for cls, p in zip(classes, proba) if cls in valid_main}
        pred_main.append(max(filtered, key=filtered.get) if filtered else classes[proba.argmax()])
    df_unclassified["Pred_Main_Category"] = pred_main

    # Update to Google Sheet
    updates = []
    for i, row in df_unclassified.iterrows():
        updates.append({'range': f'X{i + 2}', 'values': [[row["Pred_Main_Category"]]]})
        updates.append({'range': f'Y{i + 2}', 'values': [[row["Pred_Area"]]]})
        updates.append({'range': f'Z{i + 2}', 'values': [[row["Pred_Category"]]]})

    try:
        sheet_data.batch_update(updates)
    except Exception as e:
        return jsonify({"error": str(e)})

    return jsonify({"message": f"Updated {len(df_unclassified)} rows."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
