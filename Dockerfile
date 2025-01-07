FROM python:3.9.19-slim
WORKDIR /marmoset_classification
COPY . /marmoset_classification
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]