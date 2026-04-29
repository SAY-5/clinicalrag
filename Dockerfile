FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY clinicalrag ./clinicalrag
RUN pip install --no-cache-dir -e . && useradd -m -u 1001 cr
USER cr
EXPOSE 8000
ENTRYPOINT ["clinicalrag", "serve", "--host", "0.0.0.0"]
