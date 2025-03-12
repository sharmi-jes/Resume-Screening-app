from flask import Flask, request, render_template, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os
import PyPDF2
import docx
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("D:/RESUME ML PROJECTS/Resume Screening App/notebooks/model.pkl")
preprocessor = joblib.load("D:/RESUME ML PROJECTS/Resume Screening App/notebooks/tfidf_vectorizer.pkl")
label_encoder=joblib.load(r"D:\RESUME ML PROJECTS\Resume Screening App\notebooks\label_encoder.pkl")


@app.route('/')
def index():
    return render_template('index.html')


# Function to extract text from uploaded files
def extract_data_from_files(file):
    text = ""
    filename = file.filename.lower()

    if filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            text += page_text + "\n" if page_text else ""

    elif filename.endswith('.docx'):
        doc = docx.Document(file)
        text = " ".join([para.text for para in doc.paragraphs])

    elif filename.endswith(".txt"):
        text = file.read().decode("utf-8", errors="ignore")

    return text.strip()


@app.route('/predict', methods=['POST','GET'])
def predict_data():
    if request.method=="GET":
        return render_template('home.html')
    else:

     if 'resume-file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

     file = request.files['resume-file']

     if file.filename.strip() == "":
        return jsonify({"error": "No file selected"})

     text = extract_data_from_files(file)
    
     if not text:
        return jsonify({"error": "No data extracted from resume"})

    # Debugging: Print extracted text
     print(f"Extracted Text: {text}")

    # Transform text using the preprocessor (TF-IDF vectorizer)
     tfidf = preprocessor.transform([text])
    
    # Debugging: Print TF-IDF shape
     print(f"TF-IDF Shape: {tfidf.shape}")
    
    # Predict the role based on the model
     preds = model.predict(tfidf)[0]

    # Debugging: Print predicted role
     print(f"Predicted Role: {preds}")

     predicted_role=label_encoder.inverse_transform([preds])[0]

    # Return the result
    return render_template('home.html', predicted_role=predicted_role)


if __name__ == "__main__":
    app.run(debug=True)

