FROM python:3.9-slim

WORKDIR /opt/program

COPY inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference/ .

# Windows CRLF -> Linux LF (fix bash\r)
RUN sed -i 's/\r$//' serve

# Ensure executable inside Linux image
RUN chmod +x serve

EXPOSE 8080

ENTRYPOINT ["/opt/program/serve"]
