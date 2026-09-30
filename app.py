from flask import Flask, render_template, request, jsonify
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import os

import cv2
import numpy as np
import base64
import tensorflow as tf
import requests
from groq import Groq

app = Flask(__name__)

# =========================================================
# SQLITE HISTORY
# =========================================================
DB_PATH = Path("ai_pharmacy_history.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    conn = get_db()
    conn.execute("""CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        module TEXT NOT NULL,
        status TEXT,
        details TEXT
    )""")
    conn.commit(); conn.close()

def add_history(module, status, details):
    conn = get_db()
    conn.execute("INSERT INTO history (created_at,module,status,details) VALUES (?,?,?,?)",
                 (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), module, status,
                  json.dumps(details, ensure_ascii=False, default=str)))
    conn.commit(); conn.close()

init_database()

# =========================================================
# LOAD TRAINED TABLET AI MODEL
# =========================================================

MODEL_PATH = "tablet_model.keras"

try:
    tablet_model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Tablet AI model loaded successfully!")
except Exception as e:
    tablet_model = None
    print("⚠️ Tablet AI model could not be loaded:", e)


# =========================================================
# HOME + PAGES
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/drug-discovery")
def drug_discovery():
    return render_template("drug_discovery.html")


@app.route("/tablet-inspection")
def tablet_inspection():
    return render_template("tablet_inspection.html")


@app.route("/medication-safety")
def medication_safety():
    return render_template("medication_safety.html")


# =========================================================
# HISTORY + MOBILE CAMERA PAGES
# =========================================================

@app.route("/history")
def history_page():
    return render_template("history.html")

@app.route("/mobile-camera")
def mobile_camera_page():
    return render_template("mobile_camera.html")


# =========================================================
# DRUG DISCOVERY API
# =========================================================

@app.route("/api/screen-drugs", methods=["POST"])
def screen_drugs():

    try:
        data = request.get_json()

        disease = data.get("disease", "").strip()
        target = data.get("target", "").strip()

        if not disease or not target:
            return jsonify({
                "success": False,
                "error": "Please select disease and target."
            }), 400

        # -------------------------------------------------
        # REAL PUBCHEM COMPOUNDS
        # -------------------------------------------------

        # Demo mapping:
        # In a future version this can be replaced by
        # a real disease-target compound dataset/model.

        compound_map = {

            "Cancer": [
                "gefitinib",
                "erlotinib",
                "osimertinib",
                "lapatinib"
            ],

            "Diabetes": [
                "metformin",
                "glipizide",
                "pioglitazone",
                "sitagliptin"
            ],

            "Alzheimer": [
                "donepezil",
                "rivastigmine",
                "galantamine",
                "memantine"
            ],

            "Hypertension": [
                "amlodipine",
                "losartan",
                "atenolol",
                "enalapril"
            ]
        }

        compounds = compound_map.get(
            disease,
            [
                "aspirin",
                "ibuprofen",
                "paracetamol",
                "metformin"
            ]
        )

        candidates = []

        # -------------------------------------------------
        # FETCH DATA FROM PUBCHEM
        # -------------------------------------------------

        for compound_name in compounds:

            url = (
                "https://pubchem.ncbi.nlm.nih.gov/"
                "rest/pug/compound/name/"
                + requests.utils.quote(compound_name)
                + "/property/"
                "MolecularFormula,"
                "MolecularWeight,"
                "CanonicalSMILES,"
                "IsomericSMILES/"
                "JSON"
            )

            try:

                response = requests.get(
                    url,
                    timeout=10
                )

                if response.status_code != 200:
                    continue

                result = response.json()

                properties = result[
                    "PropertyTable"
                ][
                    "Properties"
                ][0]

                cid = properties.get(
                    "CID"
                )

                molecular_formula = properties.get(
                    "MolecularFormula",
                    "N/A"
                )

                molecular_weight = properties.get(
                    "MolecularWeight",
                    "N/A"
                )

                smiles = properties.get(
                    "ConnectivitySMILES",
                    properties.get(
                        "CanonicalSMILES",
                        properties.get(
                            "SMILES",
                            "N/A"
                        )
                    )
                )

                candidates.append({

                    "name":
                        compound_name.title(),

                    "cid":
                        cid,

                    "formula":
                        molecular_formula,

                    "molecular_weight":
                        molecular_weight,

                    "smiles":
                        smiles,

                    "source":
                        "PubChem",

                    "target":
                        target

                })

            except Exception as compound_error:

                print(
                    "PUBCHEM ERROR:",
                    compound_name,
                    compound_error
                )

        # -------------------------------------------------
        # NO RESULTS
        # -------------------------------------------------

        if not candidates:

            return jsonify({

                "success": False,

                "error":
                    "Could not retrieve compound data from PubChem."

            }), 502

        # -------------------------------------------------
        # SAVE HISTORY
        # -------------------------------------------------
        add_history("Drug Discovery", "SEARCH_COMPLETED", {
            "disease": disease, "target": target,
            "source": "PubChem", "candidate_count": len(candidates),
            "candidates": candidates
        })

        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return jsonify({

            "success": True,

            "disease":
                disease,

            "target":
                target,

            "source":
                "PubChem",

            "candidate_count":
                len(candidates),

            "candidates":
                candidates

        })

    except Exception as e:

        print(
            "DRUG DISCOVERY ERROR:",
            e
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# =========================================================
# GROQ AI DRUG DISCOVERY ASSISTANT
# =========================================================

@app.route("/api/gemini-drug", methods=["POST"])
def gemini_drug():
    """
    Backward-compatible endpoint name so the existing
    drug_discovery.html continues to work.

    Gemini has been removed from the backend.
    This endpoint now uses Groq AI.
    """

    try:
        data = request.get_json() or {}
        user_query = data.get("query", "").strip()

        if not user_query:
            return jsonify({
                "success": False,
                "error": "Please enter your question."
            }), 400

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            return jsonify({
                "success": False,
                "error": "Groq API key is not configured. Set GROQ_API_KEY first."
            }), 500

        client = Groq(api_key=api_key)

        system_prompt = """
You are the AI assistant inside an educational exhibition project
called "AI Pharmacy".

The user may ask about drug discovery, pharmaceutical compounds,
molecular targets, diseases, drug mechanisms, medicinal chemistry,
or related pharmaceutical research.

Give a clear, structured educational answer.

When relevant, explain:
1. Disease or condition
2. Molecular target
3. Drug/compound examples
4. Mechanism of action
5. Important molecular information
6. Drug-discovery relevance

Important safety rules:
- Do not claim that you discovered a new drug.
- Do not provide personalized medical advice or treatment instructions.
- Do not tell a specific patient which medicine to take.
- If the answer involves factual drug information, advise verification
  against authoritative sources such as PubChem, FDA labeling,
  or professional medical references.
"""

        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_query
                }
            ],
            temperature=0.2
        )

        answer = completion.choices[0].message.content

        if not answer:
            answer = "AI did not return a text response."

        add_history("Drug Discovery", "AI_QUERY", {
            "query": user_query,
            "source": "Groq AI",
            "model": "openai/gpt-oss-120b"
        })

        return jsonify({
            "success": True,
            "query": user_query,
            "answer": answer,
            "source": "Groq AI",
            "model": "openai/gpt-oss-120b"
        })

    except Exception as e:
        print("GROQ AI ERROR:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# Clean endpoint for the updated frontend.
# The old endpoint above is kept so the current HTML does not break.
@app.route("/api/ai-drug", methods=["POST"])
def ai_drug():
    return gemini_drug()


# =========================================================
# TABLET AI PREDICTION
# =========================================================

def predict_tablet_quality(cropped_tablet):

    if tablet_model is None:
        return {
            "status": "AI_MODEL_NOT_LOADED",
            "confidence": 0
        }

    try:

        # Resize according to training size
        resized = cv2.resize(
            cropped_tablet,
            (224, 224)
        )

        # OpenCV = BGR
        # TensorFlow model expects RGB
        rgb_image = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        # Convert to float
        image_array = np.array(
            rgb_image,
            dtype=np.float32
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Model prediction
        prediction = tablet_model.predict(
            image_array,
            verbose=0
        )[0][0]

        # IMPORTANT:
        # During training:
        # defective = 0
        # normal = 1
        #
        # sigmoid output:
        # near 0 -> defective
        # near 1 -> normal

        if prediction >= 0.5:

            status = "NORMAL"

            confidence = prediction * 100

        else:

            status = "DEFECTIVE"

            confidence = (1 - prediction) * 100

        return {
            "status": status,
            "confidence": round(
                float(confidence),
                2
            )
        }

    except Exception as e:

        print("AI PREDICTION ERROR:", e)

        return {
            "status": "AI_ERROR",
            "confidence": 0
        }


# =========================================================
# TABLET INSPECTION API
# =========================================================

@app.route("/api/inspect-tablet", methods=["POST"])
def inspect_tablet():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "success": False,
                "error": "No JSON data received."
            }), 400

        image_data = data.get("image")

        if not image_data:

            return jsonify({
                "success": False,
                "error": "No image received."
            }), 400

        # Remove data:image/jpeg;base64,...
        if "," in image_data:

            image_data = image_data.split(
                ",",
                1
            )[1]

        # Decode Base64
        image_bytes = base64.b64decode(
            image_data
        )

        # Convert bytes -> NumPy
        np_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        # Decode image
        image = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )

        if image is None:

            return jsonify({
                "success": False,
                "error": "Could not decode image."
            }), 400

        # =====================================================
        # BASIC IMAGE INFORMATION
        # =====================================================

        height, width = image.shape[:2]

        image_area = width * height

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        brightness = float(
            np.mean(gray)
        )

        sharpness = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

        # =====================================================
        # TABLET DETECTION
        # =====================================================

        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        _, threshold = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        threshold = cv2.morphologyEx(
            threshold,
            cv2.MORPH_CLOSE,
            kernel
        )

        threshold = cv2.morphologyEx(
            threshold,
            cv2.MORPH_OPEN,
            kernel
        )

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # =====================================================
        # FIND LARGEST TABLET-LIKE OBJECT
        # =====================================================

        detected_contour = None

        largest_area = 0

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            # Ignore tiny objects
            if area < 1500:
                continue

            # Ignore object covering almost whole frame
            if area > image_area * 0.80:
                continue

            if area > largest_area:

                largest_area = area

                detected_contour = contour

        # =====================================================
        # NO TABLET
        # =====================================================

        if detected_contour is None:

            return jsonify({

                "success": True,

                "result": "NO_TABLET",

                "message":
                    "No tablet-like object detected.",

                "brightness":
                    round(
                        brightness,
                        2
                    ),

                "sharpness":
                    round(
                        sharpness,
                        2
                    ),

                "image_width":
                    width,

                "image_height":
                    height
            })

        # =====================================================
        # TABLET BOUNDING BOX
        # =====================================================

        x, y, w, h = cv2.boundingRect(
            detected_contour
        )

        aspect_ratio = (
            w / float(h)
        )

        perimeter = cv2.arcLength(
            detected_contour,
            True
        )

        if perimeter > 0:

            circularity = (
                4 * np.pi * largest_area
            ) / (
                perimeter * perimeter
            )

        else:

            circularity = 0

        area_ratio = (
            largest_area /
            image_area
        ) * 100

        # =====================================================
        # TABLET ROI
        # =====================================================

        roi = gray[
            y:y+h,
            x:x+w
        ]

        if roi.size > 0:

            surface_mean = float(
                np.mean(roi)
            )

            surface_std = float(
                np.std(roi)
            )

        else:

            surface_mean = 0

            surface_std = 0

        # =====================================================
        # IMAGE QUALITY CHECK
        # =====================================================

        review_reasons = []

        if brightness < 35:

            review_reasons.append(
                "Image is too dark."
            )

        if brightness > 235:

            review_reasons.append(
                "Image is overexposed."
            )

        if sharpness < 50:

            review_reasons.append(
                "Image may be blurry."
            )

        if aspect_ratio < 0.30:

            review_reasons.append(
                "Detected object shape is unusual."
            )

        if aspect_ratio > 3.50:

            review_reasons.append(
                "Detected object shape is unusual."
            )

        # =====================================================
        # AI QUALITY PREDICTION
        # =====================================================

        cropped_tablet = image[
            y:y+h,
            x:x+w
        ]

        ai_result = predict_tablet_quality(
            cropped_tablet
        )

        # =====================================================
        # FINAL RESULT
        # =====================================================

        if len(review_reasons) > 0:

            result = "REVIEW"

        else:

            result = "TABLET_DETECTED"

        # =====================================================
        # SAVE HISTORY
        # =====================================================
        add_history("Tablet Inspection", ai_result["status"], {
            "result": result,
            "ai_status": ai_result["status"],
            "ai_confidence": ai_result["confidence"],
            "reasons": review_reasons,
            "brightness": round(brightness, 2),
            "sharpness": round(sharpness, 2)
        })

        # =====================================================
        # RESPONSE
        # =====================================================

        return jsonify({

            "success": True,

            "result": result,

            "message":
                "Tablet-like object detected.",

            # Bounding box
            "x": int(x),

            "y": int(y),

            "width": int(w),

            "height": int(h),

            # Shape
            "area":
                round(
                    float(largest_area),
                    2
                ),

            "area_ratio":
                round(
                    float(area_ratio),
                    2
                ),

            "aspect_ratio":
                round(
                    float(aspect_ratio),
                    2
                ),

            "circularity":
                round(
                    float(circularity),
                    3
                ),

            # Image quality
            "brightness":
                round(
                    brightness,
                    2
                ),

            "sharpness":
                round(
                    sharpness,
                    2
                ),

            "surface_mean":
                round(
                    surface_mean,
                    2
                ),

            "surface_std":
                round(
                    surface_std,
                    2
                ),

            # Image dimensions
            "image_width":
                width,

            "image_height":
                height,

            # Existing review reasons
            "reasons":
                review_reasons,

            # =================================================
            # AI RESULT
            # =================================================

            "ai_status":
                ai_result["status"],

            "ai_confidence":
                ai_result["confidence"]

        })

    except Exception as e:

        print(
            "TABLET INSPECTION ERROR:",
            e
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500

@app.route("/api/check-medications", methods=["POST"])
def check_medications():

    try:
        data = request.get_json() or {}

        medicine1 = data.get("medicine1", "").strip().lower()
        medicine2 = data.get("medicine2", "").strip().lower()
        age = int(data.get("age", 0))

        if not medicine1 or not medicine2:
            return jsonify({
                "success": False,
                "error": "Please enter both medicines."
            }), 400

        if age <= 0:
            return jsonify({
                "success": False,
                "error": "Please enter a valid age."
            }), 400

        API_URL = "https://api.fda.gov/drug/label.json"

        def get_drug_label(medicine):
            searches = [
                f'openfda.generic_name:"{medicine}"',
                f'openfda.brand_name:"{medicine}"'
            ]

            for search_query in searches:
                try:
                    response = requests.get(
                        API_URL,
                        params={"search": search_query, "limit": 1},
                        timeout=10
                    )

                    if response.status_code == 200:
                        records = response.json().get("results", [])
                        if records:
                            return records[0]
                except requests.exceptions.RequestException:
                    continue

            return None

        def extract_label_info(label):
            if not label:
                return {
                    "found": False,
                    "drug_interactions": [],
                    "warnings": [],
                    "contraindications": [],
                    "boxed_warning": []
                }

            return {
                "found": True,
                "drug_interactions": label.get("drug_interactions", []),
                "warnings": label.get("warnings", []),
                "contraindications": label.get("contraindications", []),
                "boxed_warning": label.get("boxed_warning", [])
            }

        label1 = get_drug_label(medicine1)
        label2 = get_drug_label(medicine2)

        info1 = extract_label_info(label1)
        info2 = extract_label_info(label2)

        pairwise_evidence = []

        def search_pair(first_medicine, second_medicine):
            search_query = (
                f'openfda.generic_name:"{first_medicine}" '
                f'AND drug_interactions:"{second_medicine}"'
            )

            try:
                response = requests.get(
                    API_URL,
                    params={"search": search_query, "limit": 5},
                    timeout=10
                )

                if response.status_code != 200:
                    return

                records = response.json().get("results", [])

                for record in records:
                    interactions = record.get("drug_interactions", [])

                    for interaction in interactions:
                        text = str(interaction).strip()

                        if (
                            second_medicine.lower() in text.lower()
                            and text not in pairwise_evidence
                        ):
                            pairwise_evidence.append(text)

            except requests.exceptions.RequestException as pair_error:
                print("OPENFDA PAIR SEARCH ERROR:", pair_error)

        search_pair(medicine1, medicine2)
        search_pair(medicine2, medicine1)

        if pairwise_evidence:
            risk_status = "LABEL_INTERACTION_FOUND"
            risk_title = "FDA Label Interaction Evidence Found"
            risk_reason = (
                "The retrieved FDA labeling data contains drug-interaction "
                "information mentioning this medicine pair."
            )
        elif not info1["found"] and not info2["found"]:
            risk_status = "DRUGS_NOT_FOUND"
            risk_title = "Drug Labels Not Found"
            risk_reason = (
                "No matching openFDA drug label was found for either medicine."
            )
        elif not info1["found"]:
            risk_status = "MEDICINE_1_NOT_FOUND"
            risk_title = "First Medicine Not Found"
            risk_reason = (
                "No matching openFDA label was found for the first medicine."
            )
        elif not info2["found"]:
            risk_status = "MEDICINE_2_NOT_FOUND"
            risk_title = "Second Medicine Not Found"
            risk_reason = (
                "No matching openFDA label was found for the second medicine."
            )
        else:
            risk_status = "NO_PAIRWISE_EVIDENCE"
            risk_title = "No Pairwise Label Evidence Found"
            risk_reason = (
                "No pairwise interaction statement was found in the retrieved "
                "FDA label data. This does not prove that the combination is risk-free."
            )

        if age < 18:
            age_note = (
                "Patient age is below 18. Pediatric medication use requires "
                "appropriate professional review."
            )
        elif age >= 65:
            age_note = (
                "Patient age is 65 or older. Age-related medication risks "
                "should be reviewed by a healthcare professional."
            )
        else:
            age_note = (
                "Age-specific information from the retrieved labels is not "
                "automatically interpreted by this prototype. Review the official label."
            )

        add_history("Medication Safety", risk_status, {
            "medicine1": medicine1,
            "medicine2": medicine2,
            "age": age,
            "risk_status": risk_status,
            "risk_title": risk_title,
            "reason": risk_reason,
            "age_note": age_note,
            "source": "openFDA Drug Labeling API",
            "pairwise_evidence": pairwise_evidence[:3]
        })

        return jsonify({
            "success": True,
            "medicine1": medicine1,
            "medicine2": medicine2,
            "age": age,
            "risk_status": risk_status,
            "risk_title": risk_title,
            "reason": risk_reason,
            "age_note": age_note,
            "source": "openFDA Drug Labeling API",
            "medicine1_found": info1["found"],
            "medicine2_found": info2["found"],
            "medicine1_label": {
                "drug_interactions": info1["drug_interactions"][:2],
                "warnings": info1["warnings"][:2],
                "contraindications": info1["contraindications"][:2],
                "boxed_warning": info1["boxed_warning"][:1]
            },
            "medicine2_label": {
                "drug_interactions": info2["drug_interactions"][:2],
                "warnings": info2["warnings"][:2],
                "contraindications": info2["contraindications"][:2],
                "boxed_warning": info2["boxed_warning"][:1]
            },
            "pairwise_evidence": pairwise_evidence[:3]
        })

    except ValueError:
        return jsonify({
            "success": False,
            "error": "Age must be a number."
        }), 400

    except requests.exceptions.RequestException:
        return jsonify({
            "success": False,
            "error": (
                "Unable to connect to openFDA. "
                "Please check your internet connection."
            )
        }), 503

    except Exception as e:
        print("MEDICATION SAFETY ERROR:", e)

        return jsonify({
            "success": False,
            "error": (
                "Something went wrong while checking medication information."
            )
        }), 500

# =========================================================
# HISTORY APIs
# =========================================================

@app.route("/api/history")
def api_history():
    conn = get_db()
    rows = conn.execute("SELECT id,created_at,module,status,details FROM history ORDER BY id DESC LIMIT 200").fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])

@app.route("/api/history/stats")
def api_history_stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
    tablet = conn.execute("SELECT COUNT(*) FROM history WHERE module='Tablet Inspection'").fetchone()[0]
    medication = conn.execute("SELECT COUNT(*) FROM history WHERE module='Medication Safety'").fetchone()[0]
    drugs = conn.execute("SELECT COUNT(*) FROM history WHERE module='Drug Discovery'").fetchone()[0]
    conn.close()
    return jsonify({"total":total,"tablet_inspections":tablet,"medication_checks":medication,"drug_searches":drugs})

@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    conn = get_db()
    conn.execute("DELETE FROM history")
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "History cleared successfully."})

# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )