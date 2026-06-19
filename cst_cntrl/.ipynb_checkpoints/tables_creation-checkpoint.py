from google.cloud import bigquery

# Initialize a BigQuery client
client = bigquery.Client()

# Define the dataset ID
dataset_id = 'ca-sbox-es-aiml-demo-444.demo'

# Define the schema for the CST_CNTRL_API_WRPR table
schema_cst_cntrl_api_wrpr = [
    bigquery.SchemaField("PROCESS_ID", "INTEGER", mode="REQUIRED", description="This is Unique ID for identifying the Process"),
    bigquery.SchemaField("PROCESS_NM", "STRING", mode="REQUIRED", description="This is the Process name which uniquely identifies the Process where the LLM or API is going to be called"),
    bigquery.SchemaField("PROCESS_TYP", "STRING", mode="REQUIRED", description="This identifies how the process is getting called Online /Batch for LLM ,if they dont have batch it considered Online"),
    bigquery.SchemaField("THRESHOLD", "INTEGER", mode="REQUIRED", description="This is for threshold to consider the outset a process can consider For example if 1000 request s per day and threshold is 10 then it is 1100 is max"),
    bigquery.SchemaField("FREQUENCY", "STRING", mode="REQUIRED", description="The frequency of the Process (daily,Weekly,monthly)"),
    bigquery.SchemaField("ACTIVE", "STRING", mode="REQUIRED", description="This indicator will make sure process is active or not (Y/N)"),
    bigquery.SchemaField("AVG_REQUEST", "INTEGER", mode="REQUIRED", description="This is the Request API by the frequency to check"),
    bigquery.SchemaField("CREAT_USR_ID", "STRING", mode="REQUIRED", description="This is create user or process which has created this record at the begining"),
    bigquery.SchemaField("CREAT_TS", "DATETIME", mode="REQUIRED", description="This is create datetime or process which has created this record at the begining"),
    bigquery.SchemaField("LST_UPDT_USR_ID", "STRING", mode="NULLABLE", description="This is user or process which has created this record or Updated"),
    bigquery.SchemaField("LST_UPDT_TS", "DATETIME", mode="NULLABLE", description="This is create datetime or process which has created this record or Updated")
]

# Define the schema for the CST_CNTRL_API_WRPR_LOG table
schema_cst_cntrl_api_wrpr_log = [
    bigquery.SchemaField("PROCESS_ID", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("PROCESS_NM", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("PROCESS_START_TIME", "DATETIME", mode="REQUIRED"),
    bigquery.SchemaField("PROCESS_END_TIME", "DATETIME", mode="REQUIRED"),
    bigquery.SchemaField("STATUS", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("ERROR", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("CREAT_USR_ID", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("CREAT_TS", "DATETIME", mode="REQUIRED"),
    bigquery.SchemaField("LST_UPDT_USR_ID", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("LST_UPDT_TS", "DATETIME", mode="NULLABLE")
]

# Define the schema for the CST_CNTRL_WRPR_API_CNTR_CHK table
schema_cst_cntrl_wrpr_api_cntr_chk = [
    bigquery.SchemaField("PROCESS_ID", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("PROCESS_CNTR", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("CREAT_USR_ID", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("CREAT_TS", "DATETIME", mode="REQUIRED"),
    bigquery.SchemaField("LST_UPDT_USR_ID", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("LST_UPDT_TS", "DATETIME", mode="NULLABLE")
]

# Create the tables
table_cst_cntrl_api_wrpr = bigquery.Table(f"{dataset_id}.CST_CNTRL_API_WRPR", schema=schema_cst_cntrl_api_wrpr)
table_cst_cntrl_api_wrpr_log = bigquery.Table(f"{dataset_id}.CST_CNTRL_API_WRPR_LOG", schema=schema_cst_cntrl_api_wrpr_log)
table_cst_cntrl_wrpr_api_cntr_chk = bigquery.Table(f"{dataset_id}.CST_CNTRL_WRPR_API_CNTR_CHK", schema=schema_cst_cntrl_wrpr_api_cntr_chk)

# Make API requests to create the tables
client.create_table(table_cst_cntrl_api_wrpr)
client.create_table(table_cst_cntrl_api_wrpr_log)
client.create_table(table_cst_cntrl_wrpr_api_cntr_chk)

print("Tables created successfully.")