import psycopg2
from dotenv import load_dotenv
import os
import csv
from urllib.parse import quote
import json
# from invoke_claude import filter_data


# Load environment variables from .env file
load_dotenv(dotenv_path=".env", override=True)

driver = os.getenv("DRIVER")
host = os.getenv("HOST")
port = os.getenv("PORT")
database = os.getenv("DATABASE")
username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")

connection_params = {
    'dbname': database,
    'user': username,
    'password': password,
    'host': host,
    'port': port,
}


def execute_query(query, single_item = False):
    response = None
    try:
        conn = psycopg2.connect(**connection_params)
        cursor = conn.cursor()
        cursor.execute(query)
    
        if single_item:
            response = cursor.fetchone()
        else:
            response = cursor.fetchall()
    
    except Exception as e:
        print(f"Error connecting to the PostgreSQL database: {e}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    return response


def split_test_dataset(test_dataset_path, uuid_output_path, alarm_output_path):
    # Load the test dataset
    with open(test_dataset_path, mode="r", encoding="utf-8") as file:
        test_dataset = json.load(file)
    
    uuid_test_dataset = []
    alarm_test_dataset = []

    # Iterate through the dataset
    for item in test_dataset:
        query = item[0]["en"]  # English query
        source_uris = item[1]  # List of URIs
        encoded_uris = [quote(uri) for uri in source_uris]

        # Check if the answers belong to UUID or Alarm guide books
        if any(check_if_answer_in_uuid(encoded_uri) for encoded_uri in encoded_uris):
            uuid_test_dataset.append({"query": query, "source_uri": source_uris})
        if any(check_if_answer_in_alarm(encoded_uri) for encoded_uri in encoded_uris):
            alarm_test_dataset.append({"query": query, "source_uri": source_uris})

    # Save the UUID test dataset
    with open(uuid_output_path, mode="w", encoding="utf-8") as uuid_file:
        writer = csv.DictWriter(uuid_file, fieldnames=["query", "source_uri"])
        writer.writeheader()
        writer.writerows(uuid_test_dataset)

    # Save the Alarm test dataset
    with open(alarm_output_path, mode="w", encoding="utf-8") as alarm_file:
        writer = csv.DictWriter(alarm_file, fieldnames=["query", "source_uri"])
        writer.writeheader()
        writer.writerows(alarm_test_dataset)

    print(f"UUID test dataset saved to {uuid_output_path}")
    print(f"Alarm test dataset saved to {alarm_output_path}")


def check_if_answer_in_uuid(encoded_uri:str) -> str:
    query = f"""
    SELECT source_uri
    FROM v9__chatbot_documents
    WHERE source_uri like '%58548175-ccef-4d6a-987c-f597b7d4d225%{encoded_uri}%' 
    """
    response = execute_query(query)
    return len(response) > 0

def check_if_answer_in_alarm(encoded_uri:str) -> str:
    query = f"""
    SELECT source_uri
    FROM v9__chatbot_documents
    WHERE source_uri like '%me_c_mk2%{encoded_uri}%' 
    """
    response = execute_query(query)
    return len(response) > 0
# origin
# def get_parent_chunk(encoded_uri:str) -> str:
#     query = f"""
#     SELECT document_contents
#     FROM v9__chatbot_documents
#     WHERE source_uri like '%58548175-ccef-4d6a-987c-f597b7d4d225%{encoded_uri}%' OR source_uri like '%me_c_mk2%{encoded_uri}%'
#     LIMIT 1
#     """
#     response = execute_query(query, single_item=True)
#     return response[0]

# brevico
# def get_parent_chunk(encoded_uri:str) -> str:
#     query = f"""
#     SELECT document_contents
#     FROM v9__chatbot_documents
#     WHERE source_uri like '%13cd045f-fa1e-4f83-b91b-e28244ae415a%{encoded_uri}%'
#     LIMIT 1
#     response = execute_query(query, single_item=True)
#     return response[0]

# Torm success
def get_parent_chunk(encoded_uri:str) -> str:
    query = f"""
    SELECT document_contents
    FROM v9__chatbot_documents
    WHERE source_uri like '%58548175-ccef-4d6a-987c-f597b7d4d225%{encoded_uri}%'
    LIMIT 1
    """
    response = execute_query(query, single_item=True)
    return response[0]

def merge_csv_files(file1, file2, output_file):
    # 打开输出文件
    with open(output_file, mode="w", encoding="utf-8", newline="") as outfile:
        writer = None

        # 处理第一个文件
        with open(file1, mode="r", encoding="utf-8") as infile1:
            reader1 = csv.DictReader(infile1)
            if writer is None:
                # 初始化 DictWriter，并写入表头
                writer = csv.DictWriter(outfile, fieldnames=reader1.fieldnames)
                writer.writeheader()
            for row in reader1:
                writer.writerow(row)

        # 处理第二个文件
        with open(file2, mode="r", encoding="utf-8") as infile2:
            reader2 = csv.DictReader(infile2)
            for row in reader2:
                writer.writerow(row)

    print(f"Files merged successfully into {output_file}")

if __name__ == "__main__":
    # test_dataset_path = "test_dataset.json"
    # uuid_output_path = "uuid_test_dataset.csv"
    # alarm_output_path = "alarm_test_dataset.csv"

    # split_test_dataset(test_dataset_path, uuid_output_path, alarm_output_path)

    get_parent_chunk('M5200106-11.html')

