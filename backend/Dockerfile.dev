FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

WORKDIR '/back'

RUN apt-get update && \
apt-get install -y libgl1 libglib2.0-0 build-essential gcc python3-dev && \
apt-get clean

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--reload"]