# app.py
from flask import Flask, request, jsonify
import numpy as np
from lime.lime_text import LimeTextExplainer
from utils import load_config, load_tokenizer, load_label_classes, load_sentiment_model, prepare_input

# ========== Load Components ==========
config = load_config()
host = config["app"]["host"]
port = config["app"]["internal_port"]
max_len = config["app"]["max_len"]

tokenizer = load_tokenizer()
label_classes = np.array(["negative", "neutral", "positive"])
model = load_sentiment_model()

explainer = LimeTextExplainer(class_names=label_classes.tolist())

# ========== App Init ==========
app = Flask(__name__)

def predict_proba(texts):
    padded = prepare_input(texts, tokenizer, max_len)
    return model.predict(padded)

# ========== Prediction Endpoint ==========
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No input text provided."}), 400

    text = data["text"].strip()
    if not text or len(text) < 3:
        return jsonify({"error": "Input text too short."}), 400

    sequence = tokenizer.texts_to_sequences([text])
    tokens = sequence[0]
    unique_tokens = set(tokens)

    if len(tokens) < 3 or (len(unique_tokens) <= 1 and tokens.count(tokens[0]) > 1):
        return jsonify({
            "prediction": "Garbage",
            "confidence": 0.0,
            "top_contributing_words": [],
            "note": "Input flagged as garbage."
        })

    padded = prepare_input([text], tokenizer, max_len)
    predictions = model.predict(padded)[0]
    label_idx = int(np.argmax(predictions))
    label = label_classes[label_idx]
    confidence = float(predictions[label_idx])

    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_proba,
        num_features=10,
        top_labels=1
    )
    top_expl = explanation.as_list(label=explanation.top_labels[0])

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 4),
        "top_contributing_words": top_expl
    })

# ========== UI Test ==========
@app.route("/test-ui")
def test_ui():
    return '''
    <h2>Sentiment Test</h2>
    <form onsubmit="submitForm(event)">
      <input type="text" id="text" size="60">
      <input type="submit" value="Analyze">
    </form>
    <pre id="result"></pre>
    <script>
      async function submitForm(event) {
        event.preventDefault();
        const text = document.getElementById("text").value;
        const res = await fetch("/predict", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({text})
        });
        const json = await res.json();
        document.getElementById("result").textContent = JSON.stringify(json, null, 2);
      }
    </script>
    '''

# ========== Root ==========
@app.route("/")
def home():
    return "<h3>Use /predict (POST) or /test-ui</h3>"

# ========== Run ==========
if __name__ == "__main__":
    app.run(host=host, port=port)
