import yaml
from flask import Flask, render_template_string, send_file, url_for
from gan_model import generate_image
import os

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
flask_config = config["flask"]
gan_config = config["gan"]

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>GAN Image Generator</title>
</head>
<body>
    <h1>GAN Image Generator</h1>
    <form method="POST" action="/generate">
        <button type="submit">Generate Image</button>
    </form>
    {% if image_url %}
    <h2>Generated Image:</h2>
    <img src="{{ image_url }}" alt="Generated Image" width="256">
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML, image_url=None)

@app.route('/generate', methods=['POST'])
def generate():
    generate_image()
    return render_template_string(HTML, image_url=url_for('get_image'))

@app.route('/image')
def get_image():
    img_path = gan_config.get("output_image", "output.png")
    if not os.path.exists(img_path):
        return "No image generated yet.", 404
    return send_file(img_path, mimetype='image/png')

if __name__ == "__main__":
    app.run(
        host=flask_config["host"],
        port=flask_config["port"]
    )