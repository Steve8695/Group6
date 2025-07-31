**Run Locally**
python -m venv venv
source venv/bin/activate   
pip install -r requirements.txt
uvicorn app.app:app --reload --host 0.0.0.0 --port 8502




**Run with Docker**
docker build -t rag-fastapi .
docker run -p 8502:8502 rag-fastapi
