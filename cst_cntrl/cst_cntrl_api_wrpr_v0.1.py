from google.cloud import bigquery
import requests
import pdb

class GeminiRequestHandler:
    def __init__(self, project_id, dataset_id, table_name, gemini_api_url):
        self.client = bigquery.Client()
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_name = table_name
        self.gemini_api_url = gemini_api_url

    def get_process_info(self, process_nm):
        query = f"""
        SELECT PROCESS_ID, THRESHOLD, AVG_REQUEST
        FROM `{self.project_id}.{self.dataset_id}.{self.table_name}`
        WHERE PROCESS_NM = @process_nm
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("process_nm", "STRING", process_nm)
            ]
        )
        query_job = self.client.query(query, job_config=job_config)
        results = query_job.result()
        return results

    def check_threshold(self, process_nm):
        results = self.get_process_info(process_nm)
        for row in results:
            process_id = row.PROCESS_ID
            threshold = row.THRESHOLD
            avg_request = row.AVG_REQUEST

            # Calculate the total allowed requests
            total_allowed_requests = avg_request + threshold

            # Get the current number of requests
            current_requests = self.get_current_requests(process_id)

            if current_requests > total_allowed_requests:
                return False
        return True

    def get_current_requests(self, process_id):
        query = f"""
        SELECT COUNT(*) as request_count
        FROM `{self.project_id}.{self.dataset_id}.CST_CNTRL_API_WRPR_LOG`
        WHERE PROCESS_ID = @process_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("process_id", "INTEGER", process_id)
            ]
        )
        query_job = self.client.query(query, job_config=job_config)
        results = query_job.result()
        for row in results:
            return row.request_count
        return 0

    def send_to_gemini(self, payload):
        response = requests.post(self.gemini_api_url, json=payload)
        return response.json()

    def handle_request(self, process_nm, payload):
        if self.check_threshold(process_nm):
            response = self.send_to_gemini(payload)
            return response
        else:
            return {"error": "Request threshold exceeded"}

# Usage example
if __name__ == "__main__":
    project_id = "ca-sbox-es-aiml-demo-444"
    dataset_id = "demo"
    table_name = "CST_CNTRL_API_WRPR"
    gemini_api_url = "https://gemini-llm-api.example.com/endpoint"

    handler = GeminiRequestHandler(project_id, dataset_id, table_name, gemini_api_url)
    process_nm = "Example Process"
    payload = {"data": "example data"}

    response = handler.handle_request(process_nm, payload)
    print(response)