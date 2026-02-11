import os
import json
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types
import PyPDF2
from flask_cors import CORS

# ==============================
# CONFIG
# ==============================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Note: In a production environment, use environment variables
client = genai.Client(api_key="YOUR_API_KEY")

app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==============================
# PDF PARSING
# ==============================
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text

# ==============================
# ATS LOGIC (Combined for Efficiency)
# ==============================
def get_ats_analysis(resume_text, jd_text):
    prompt = f"""
    You are an expert Applicant Tracking System (ATS) specializing in Tech Recruitment.
    
    Task: Analyze the provided Resume against the Job Description.
    
    Resume Content:
    {resume_text}
    
    Job Description:
    {jd_text}
    
    Return the analysis strictly as a JSON object with this structure:
    {{
        "match_percentage": number,
        "match_level": "Strong Fit" | "Good Fit" | "Potential Fit" | "Low Match",
        "matching_skills": [string],
        "missing_skills": [string],
        "strengths": [string],
        "suggestions": [string],
        "summary": string
    }}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return None

# ==============================
# API ROUTE
# ==============================
@app.route("/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "Resume PDF is required"}), 400

    resume_file = request.files["resume"]
    jd_text = request.form.get("job_description")

    if not jd_text:
        return jsonify({"error": "Job description is required"}), 400

    # Save PDF
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], resume_file.filename)
    resume_file.save(pdf_path)

    # 1. Extract text
    resume_text = extract_text_from_pdf(pdf_path)
    
    if not resume_text.strip():
        return jsonify({"error": "Could not extract text from PDF"}), 400

    # 2. Get Analysis from Gemini
    analysis = get_ats_analysis(resume_text, jd_text)

    if not analysis:
        return jsonify({"error": "AI analysis failed"}), 500

    return jsonify(analysis)

if __name__ == "__main__":
    app.run(debug=True, port=8080)
