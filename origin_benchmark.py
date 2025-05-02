import csv
import json
import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel
from urllib.parse import unquote


# 初始化 bge-m3 模型
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# 文件路径
test_dataset_path = "origin_test_dataset_embeddings/alarm_test_dataset.csv"
pool_path = "origin_pool_embeddings/total_pool.csv"
mix_output_csv_path = "origin_mix_incorrect_questions.csv"
top_5_dense_output_csv_path = "origin_top_5_dense_incorrect_questions.csv"
top_3_dense_output_csv_path = "origin_top_3_dense_incorrect_questions.csv"
top_5_sparse_output_csv_path = "origin_top_5_sparse_incorrect_questions.csv"
top_2_sparse_output_csv_path = "origin_top_2_sparse_incorrect_questions.csv"

# 读取测试数据集
questions = []
dense_question_embeddings = []
sparse_question_embeddings = []
answers = []
with open(test_dataset_path, mode="r", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        dense_question_embeddings.append(json.loads(row["embedding"]))
        sparse_question_embeddings.append(json.loads(row["embedding_sparse"]))
        answers.append(json.loads(row["source_uri"].replace("'", '"')))
        questions.append(row["query"])


dense_pool_embeddings = []
sparse_pool_embeddings = []
source_uris = []
with open(pool_path, mode="r", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        dense_pool_embeddings.append(json.loads(row["embedding"]))
        sparse_pool_embeddings.append(json.loads(row["embedding_sparse"]))        
        source_uris.append(row["source_uri"])




# 计算距离矩阵（余弦相似度）
dense_question_embeddings = np.array(dense_question_embeddings)
dense_pool_embeddings = np.array(dense_pool_embeddings)
distance_matrix = dense_question_embeddings @ dense_pool_embeddings.T

# 获取每个问题最近的 3 个索引
top_3_indices_dense = np.argsort(-distance_matrix, axis=1)[:, :3]  # 按相似度降序排序
# 获取每个问题最近的 5 个索引
top_5_indices_dense = np.argsort(-distance_matrix, axis=1)[:, :5] 



# calculate Matrix of Sparse

# create index matrix for sparse
top_2_indices_sparse = []
top_5_indices_sparse = []
for q_embedding in sparse_question_embeddings:
    distances = []
    for idx, p_embedding in enumerate(sparse_pool_embeddings):
        distance = model.compute_lexical_matching_score(q_embedding, p_embedding)
        distances.append((idx, distance))  # store index and distance
    # 按距离降序排列
    distances.sort(key=lambda x: x[1], reverse=True)
    # 提取前 2 项的索引
    top_2_indices = [item[0] for item in distances[:2]]
    top_2_indices_sparse.append(top_2_indices)

    # 提取前 5 项的索引
    top_5_indices = [item[0] for item in distances[:5]]
    top_5_indices_sparse.append(top_5_indices)




# 比对答案并统计得分
mix_score = 0
mix_incorrect_questions = []

top_5_dense_score=0
top_5_dense_incorrect_questions = []

top_3_dense_score=0
top_3_dense_incorrect_questions = []

top_5_sparse_score=0
top_5_sparse_incorrect_questions = []

top_2_sparse_score=0
top_2_sparse_incorrect_questions = []

for i, answer in enumerate(answers):
    question = questions[i]

    mix_top_indices = set(top_3_indices_dense[i]) | set(top_2_indices_sparse[i])  # 获取最近的 3 +2  个索引

    top_mix_source_uris = [unquote(source_uris[idx]).split("/")[-1] for idx in mix_top_indices]  # 提取对应的 source_uri

    top_5_dense_source_uris = [unquote(source_uris[idx]).split("/")[-1] for idx in set(top_5_indices_dense[i])]
        
    top_3_dense_source_uris = [unquote(source_uris[idx]).split("/")[-1] for idx in set(top_3_indices_dense[i])] 

    top_5_sparse_source_uris = [unquote(source_uris[idx]).split("/")[-1] for idx in set(top_5_indices_sparse[i])]
        
    top_2_sparse_source_uris = [unquote(source_uris[idx]).split("/")[-1] for idx in set(top_2_indices_sparse[i])]



    # 检查答案和最近的 source_uri 是否有交集

    # if all(item in top_mix_source_uris for item in answer):
    if bool(set(answer) & set(top_mix_source_uris)):
        mix_score += 1
    else:
        mix_incorrect_questions.append({"question": question, "answers": answer, "top_uris": top_mix_source_uris})

    if bool(set(answer) & set(top_5_dense_source_uris)):
        top_5_dense_score += 1
    else:
        top_5_dense_incorrect_questions.append({"question": question, "answers": answer, "top_uris": top_5_dense_source_uris})

    if bool(set(answer) & set(top_3_dense_source_uris)):
        top_3_dense_score += 1
    else:
        top_3_dense_incorrect_questions.append({"question": question, "answers": answer, "top_uris": top_3_dense_source_uris})
        
    if bool(set(answer) & set(top_5_sparse_source_uris)):
        top_5_sparse_score += 1
    else:
        top_5_sparse_incorrect_questions.append({"question": question, "answers": answer, "top_uris": top_5_sparse_source_uris})

    if bool(set(answer) & set(top_2_sparse_source_uris)):
        top_2_sparse_score += 1
    else:
        top_2_sparse_incorrect_questions.append({"question": question, "answers": answer, "top_uris": top_2_sparse_source_uris})

# 输出得分
total_questions = len(questions)
print(f"origin_mix_score: {mix_score}/{total_questions}")
print(f"origin_top_5_dense_score: {top_5_dense_score}/{total_questions}")
print(f"origin_top_3_dense_score: {top_3_dense_score}/{total_questions}")
print(f"origin_top_5_sparse_score: {top_5_sparse_score}/{total_questions}")
print(f"origin_top_2_sparse_score: {top_2_sparse_score}/{total_questions}")

# 将题目保存到 CSV 文件
pd.DataFrame(mix_incorrect_questions).to_csv(mix_output_csv_path, index=False, encoding="utf-8")
print(f"Mixed method: Incorrect questions saved to {mix_output_csv_path}")

pd.DataFrame(top_5_dense_incorrect_questions).to_csv(top_5_dense_output_csv_path, index=False, encoding="utf-8")
print(f"Top 5 Dense method: Incorrect questions saved to {top_5_dense_output_csv_path}")

pd.DataFrame(top_3_dense_incorrect_questions).to_csv(top_3_dense_output_csv_path, index=False, encoding="utf-8")
print(f"Top 3 Dense method: Incorrect questions saved to {top_3_dense_output_csv_path}")

pd.DataFrame(top_5_sparse_incorrect_questions).to_csv(top_5_sparse_output_csv_path, index=False, encoding="utf-8")
print(f"Top 5 Sparse method: Incorrect questions saved to {top_5_sparse_output_csv_path}")

pd.DataFrame(top_2_sparse_incorrect_questions).to_csv(top_2_sparse_output_csv_path, index=False, encoding="utf-8")
print(f"Top 2 Sparse method: Incorrect questions saved to {top_2_sparse_output_csv_path}")

