# GAN Flask All-in-One

## Run Locally

```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then visit: [http://localhost:8000](http://localhost:8000)

## Run with Docker

```sh
docker build -t gan-flask .
docker run -p 8000:8000 gan-flask
```

Open your browser at [http://localhost:8000](http://localhost:8000)
