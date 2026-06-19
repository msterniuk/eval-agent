from google.cloud import bigquery
import requests
import pdb
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',  # Log file name
    filemode='w'  # Overwrite the log file each time the program runs
)


class GeminiRequestHandler:
    def __init__(self, prompt, process_nm, process_type):
        self.client = bigquery.Client()
        self.prompt = prompt
        self.process_nm = process_nm
        self.process_type = process_type
    
    def display_values(self):
        logging.debug(f"Prompt: {prompt}")
        logging.debug(f"Process Name: {process_nm}")
        logging.debug(f"Process Type: {process_type}") 
        
    
    def query_table(self):
        self.display_values()
        
        query = """
        SELECT *
        FROM `ca-app-giad-txt-analy-dev-444.ETL_WORK_DB.CST_CNTRL_API_WRPR`
        WHERE LOWER(PROCESS_NM) = LOWER(@process_nm) AND LOWER(PROCESS_TYP) = LOWER(@process_type)
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("process_nm", "STRING", self.process_nm),
                bigquery.ScalarQueryParameter("process_type", "STRING", self.process_type)
            ]
        )
        
        query_job = self.client.query(query, job_config=job_config)
        
        results = query_job.result()
        
        for row in results:
            print(row)
            logging.debug(f"fetching cost control checks: {row}")

    
