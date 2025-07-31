from flask import Flask, render_template, request, send_from_directory
from inference import generate_image
import os
import uuid

app = Flask(__name__)
OUTPUT_DIR = "generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.form["prompt"]
    image_name = f"{uuid.uuid4().hex}.png"
    save_path = os.path.join(OUTPUT_DIR, image_name)
    generate_image(prompt, save_path)
    return image_name

@app.route("/generated/<filename>")
def send_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
