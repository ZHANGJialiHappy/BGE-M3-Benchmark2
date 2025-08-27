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
