FROM europe-west1-docker.pkg.dev/spanner-demo-bengal/my-docker-repo/base-image:latest

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["functions-framework", "--target=consolidator", "--source=main.py"]
