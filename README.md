BGE-M3 Embedding Benchmark

This project provides a benchmark pipeline for the BGE-M3 embedding model.
The workflow includes:

Computing embeddings for chunked text

Running an initial benchmark

Using Claude to refine and correct the evaluation

At the end, you will get a file in the project root that contains:

The questions the model answered incorrectly

The model’s answers

The correct answers

📦 Environment Setup
1. Create a virtual environment
python -m venv venv
# Activate on Linux / MacOS
source venv/bin/activate
# Activate on Windows PowerShell
venv\Scripts\activate

2. Install dependencies
pip install -r requirements.txt

🚀 Usage
1. Calculate embeddings
python calculate_finetune_embedding.py

2. Run the initial benchmark
python origin_benchmark.py

3. Refine results with Claude
python invoke_claude.py

📊 Output

After running all steps, the project root will contain a file listing:

Misanswered questions

The model’s predictions

The correct answers
