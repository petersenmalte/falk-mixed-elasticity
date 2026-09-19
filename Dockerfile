FROM dolfinx/dolfinx:stable

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY . .

CMD ["python3", "scripts/convergence_study.py"]
