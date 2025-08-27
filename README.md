# BGE-M3 Embedding Benchmark

This project provides a benchmark pipeline for the **BGE-M3 embedding model**.  
The workflow includes:  
1. Computing embeddings for chunked text  
2. Running an initial benchmark  
3. Using Claude to refine and correct the evaluation  

At the end, you will get a file in the project root that contains:  
- The questions the model answered incorrectly  
- The model’s answers  
- The correct answers  

---

## 📦 Environment Setup

### 1. Create a virtual environment
```bash
python -m venv venv
# Activate on Linux / MacOS
source venv/bin/activate
# Activate on Windows PowerShell
venv\Scripts\activate

### 2. Install dependencies
```bash
pip install -r requirements.txt

## 🚀 Usage
1. Calculate embeddings
```bash
python calculate_finetune_embedding.py

2. Run the initial benchmark
```bash
python origin_benchmark.py

3. Refine results with Claude
```bash
python invoke_claude.py

## 📊 Output
After running all steps, the project root will contain a file listing: /n

1. Misanswered questions /n

2. The model’s predictions /n

3. The correct answers


