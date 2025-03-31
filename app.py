from flask import Flask, request, render_template, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os
import PyPDF2
import docx
import joblib

app = Flask(__name__)

# Load trained model and vectorizer from the local directory
model = joblib.load(os.path.join("notebooks", "model.pkl"))
preprocessor = joblib.load(os.path.join("notebooks", "tfidf_vectorizer.pkl"))
label_encoder = joblib.load(os.path.join("notebooks", "label_encoder.pkl"))


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


@app.route('/predict', methods=['POST', 'GET'])
def predict_data():
    if request.method == "GET":
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

        # Transform text using the preprocessor (TF-IDF vectorizer)
        tfidf = preprocessor.transform([text])

        # Predict the role
        preds = model.predict(tfidf)[0]
        predicted_role = label_encoder.inverse_transform([preds])[0]

        # Return the result
        return render_template('home.html', predicted_role=predicted_role)


# Port binding for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render requires this for port binding
    app.run(host='0.0.0.0', port=port, debug=True)
