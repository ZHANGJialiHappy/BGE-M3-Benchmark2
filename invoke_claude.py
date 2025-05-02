import boto3
import json
import csv
from urllib.parse import unquote, quote
from file_utils import get_parent_chunk


def invoke_claude_3_with_text(prompt):

    client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )

    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "temperature": 0,
                "max_tokens": 3500,
                "stop_sequences": ["Human:", "\n\nHuman:", "</Answer>", "\n</Answer>"],
                "system": "give me your name",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "<Answer>"}],
                    },
                ],
            }
        ),
    )

    result = json.loads(response.get("body").read())
    output_list = result.get("content", [])

    # return output_list[0]["text"]
    return output_list[0]["text"] if output_list else {}


def generate_query_for_alarm(content):
    prompt = f"""
    I will provide you with some content. The first line of the content always starts with "Alarm" followed by an alarm_id. 

    Your task is to generate a question that includes the alarm_id and is directly related to the content provided. 
    Do not include any additional text or explanations, only return the question.

    Content:
    {content}
    """
    
    return invoke_claude_3_with_text(prompt)

def generate_query_for_alarms():
    # 读取database中下载的csv文件（source_uri，embedding_text）,已经delete了
    with open("alarm_id_source.csv", mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # 打开目标文件以追加内容
        with open("alarm_test_dataset.csv", mode='a', encoding='utf-8', newline='') as output_file:
            fieldnames = ["query", "source_uri"]
            csv_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            
            # 检查文件是否为空，如果为空则写入表头
            output_file.seek(0, 2)  # 移动到文件末尾
            if output_file.tell() == 0:
                csv_writer.writeheader()
            
            # 逐行处理输入文件
            for row in csv_reader:
                source_uri = f"['{unquote(row['source_uri']).split('/')[-1]}']"
                content = row["embedding_text"]
                query = generate_query_for_alarm(content)
                
                # 确保 query 和 source_uri 格式正确
                query = query.strip().replace("\n", " ").replace('"', "'")
                source_uri = source_uri.strip().replace("\n", " ")
                
                # 写入到目标文件
                csv_writer.writerow({"query": query, "source_uri": source_uri})

def check_wrong_answer(query, content):
    prompt = f"""
    Determine if the following content:
    {content}
    correctly answers the question: {query}.
    Respond only with "True" or "False".
    Do not include any additional text or explanations.
    """
    return invoke_claude_3_with_text(prompt)

def process_incorrect_questions(input_csv_path, output_csv_path):
    # 打开输入文件和输出文件
    with open(input_csv_path, mode="r", encoding="utf-8") as input_file, \
         open(output_csv_path, mode="w", encoding="utf-8", newline="") as output_file:

        reader = csv.DictReader(input_file)
        fieldnames = ["question", "correct_top_uris"]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        # 逐行处理输入文件
        for row in reader:
            question = row["question"]
            top_uris = json.loads(row["top_uris"].replace("'", '"'))  # 将字符串解析为列表
            correct_uris = []

            # 遍历 top_uris
            for uri in top_uris:
                content = get_parent_chunk(quote(uri))  # 调用 get_parent_chunk 获取内容
                if not content:
                    continue  # 如果 content 为空，跳过

                # 调用 check_wrong_answer 检查答案是否正确
                result = check_wrong_answer(question, content)
                if result.strip().replace("\n", " ").replace('"', "'") == "True":
                    correct_uris.append(uri)

            # 如果有正确的 URI，将其写入输出文件
            if correct_uris:
                writer.writerow({
                    "question": question,
                    "correct_top_uris": json.dumps(correct_uris)
                })
if __name__ == "__main__":
    input_csv_path = "finetune_mix_incorrect_questions.csv"
    output_csv_path = "correct_questions_and_uris.csv"
    process_incorrect_questions(input_csv_path, output_csv_path)
