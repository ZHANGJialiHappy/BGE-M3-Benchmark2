import csv
import json
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# get alarm_test_dataset embeddings
dataset_path = "test_dataset/alarm_test_dataset.csv"
output_csv_path = "origin_test_dataset_embeddings/alarm_test_dataset.csv"


questions = []
rows = []


with open(dataset_path, mode="r", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        questions.append(row["query"])
        rows.append(row) 




dense_embeddings = model.encode(questions, batch_size=12)['dense_vecs']

sparse_embeddings = model.encode(questions, return_dense=True, return_sparse=True, return_colbert_vecs=False)['lexical_weights']

with open(output_csv_path, mode="w", encoding="utf-8", newline="") as csv_file:
    fieldnames = ["query", "source_uri", "embedding", "embedding_sparse"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for i, row in enumerate(rows):
        row["embedding"] = json.dumps(dense_embeddings[i].tolist())  # dense_embedding = np.array([0.123, 0.456, 0.789])得转换成list
        sparse_embedding = sparse_embeddings[i]
        sparse_embedding = {k: float(v) for k, v in sparse_embedding.items()}
        row["embedding_sparse"] = json.dumps(sparse_embedding)  
        writer.writerow(row)


print(f"Embeddings have been written to {output_csv_path}")





# get uuid_test_dataset embeddings

dataset_path = "test_dataset/uuid_test_dataset.csv"
output_csv_path = "origin_test_dataset_embeddings/uuid_test_dataset.csv"


questions = []
rows = []


with open(dataset_path, mode="r", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        questions.append(row["query"])
        rows.append(row) 




dense_embeddings = model.encode(questions, batch_size=12)['dense_vecs']

sparse_embeddings = model.encode(questions, return_dense=True, return_sparse=True, return_colbert_vecs=False)['lexical_weights']

with open(output_csv_path, mode="w", encoding="utf-8", newline="") as csv_file:
    fieldnames = ["query", "source_uri", "embedding", "embedding_sparse"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for i, row in enumerate(rows):
        row["embedding"] = json.dumps(dense_embeddings[i].tolist())  # dense_embedding = np.array([0.123, 0.456, 0.789])得转换成list
        sparse_embedding = sparse_embeddings[i]
        sparse_embedding = {k: float(v) for k, v in sparse_embedding.items()}
        row["embedding_sparse"] = json.dumps(sparse_embedding)  
        writer.writerow(row)


print(f"Embeddings have been written to {output_csv_path}")






# get alarm_pool embeddings

dataset_path = "pool/alarm_pool.csv"
output_csv_path = "origin_pool_embeddings/alarm_pool.csv"


embedding_texts = []
rows = []


with open(dataset_path, mode="r", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        embedding_texts.append(row["embedding_text"])
        rows.append(row) 




dense_embeddings = model.encode(embedding_texts, batch_size=12)['dense_vecs']

sparse_embeddings = model.encode(embedding_texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)['lexical_weights']

with open(output_csv_path, mode="w", encoding="utf-8", newline="") as csv_file:
    fieldnames = ["source_uri", "embedding", "embedding_sparse", "embedding_text"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for i, row in enumerate(rows):
        row["embedding"] = json.dumps(dense_embeddings[i].tolist())  # dense_embedding = np.array([0.123, 0.456, 0.789])得转换成list
        sparse_embedding = sparse_embeddings[i]
        sparse_embedding = {k: float(v) for k, v in sparse_embedding.items()}
        row["embedding_sparse"] = json.dumps(sparse_embedding)  
        writer.writerow(row)


print(f"Embeddings have been written to {output_csv_path}")





# get uuid_pool embeddings

dataset_path = "pool/uuid_pool.csv"
output_csv_path = "origin_pool_embeddings/uuid_pool.csv"


embedding_texts = []
rows = []


with open(dataset_path, mode="r", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        embedding_texts.append(row["embedding_text"])
        rows.append(row) 




dense_embeddings = model.encode(embedding_texts, batch_size=12)['dense_vecs']

sparse_embeddings = model.encode(embedding_texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)['lexical_weights']

with open(output_csv_path, mode="w", encoding="utf-8", newline="") as csv_file:
    fieldnames = ["source_uri", "embedding", "embedding_sparse", "embedding_text"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for i, row in enumerate(rows):
        row["embedding"] = json.dumps(dense_embeddings[i].tolist())  # dense_embedding = np.array([0.123, 0.456, 0.789])得转换成list
        sparse_embedding = sparse_embeddings[i]
        sparse_embedding = {k: float(v) for k, v in sparse_embedding.items()}
        row["embedding_sparse"] = json.dumps(sparse_embedding)  
        writer.writerow(row)


print(f"Embeddings have been written to {output_csv_path}")






