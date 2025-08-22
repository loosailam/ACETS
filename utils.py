import os
import datetime
import pytz
import pyodbc
import logging
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
# Use DefaultAzureCredential for non-interactive authentication
from azure.identity import DefaultAzureCredential
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s',
        handlers=[
            logging.FileHandler("app.log", mode="a", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )


def load_env_variables():
    load_dotenv()
    return {
        # Speech resource
        "speech_region": os.environ.get('SPEECH_REGION'), # e.g. westus2
        "speech_key": os.environ.get('SPEECH_KEY'),
        "speech_private_endpoint": os.environ.get('SPEECH_PRIVATE_ENDPOINT'), # e.g. https://my-speech-service.cognitiveservices.azure.com/ (optional)
        "speech_resource_url": os.environ.get('SPEECH_RESOURCE_URL'), # e.g. /subscriptions/6e83d8b7-00dd-4b0a-9e98-dab9f060418b/resourceGroups/my-rg/providers/Microsoft.CognitiveServices/accounts/my-speech (optional, only used for private endpoint)
        "user_assigned_managed_identity_client_id": os.environ.get('USER_ASSIGNED_MANAGED_IDENTITY_CLIENT_ID'), # e.g. the client id of user assigned managed identity accociated to your app service (optional, only used for private endpoint and user assigned managed identity)
        # OpenAI resource
        "azure_openai_endpoint": os.environ.get('AZURE_OPENAI_ENDPOINT'), # e.g. https://my-aoai.openai.azure.com/
        "azure_openai_api_key": os.environ.get('AZURE_OPENAI_API_KEY'),
        "azure_openai_deployment_name": os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME'), # e.g. my-gpt-35-turbo-deployment
        # Cognitive search resource (optional)
        "cognitive_search_endpoint": os.environ.get('COGNITIVE_SEARCH_ENDPOINT'), # e.g. https://my-cognitive-search.search.windows.net/
        "cognitive_search_api_key": os.environ.get('COGNITIVE_SEARCH_API_KEY'),
        "cognitive_search_index_name": os.environ.get('COGNITIVE_SEARCH_INDEX_NAME'), # e.g. my-search-index
        # Customized ICE server (optional)
        "ice_server_url": os.environ.get('ICE_SERVER_URL'), # The ICE URL, e.g. turn:x.x.x.x:3478
        "ice_server_url_remote": os.environ.get('ICE_SERVER_URL_REMOTE'), # The ICE URL for remote side, e.g. turn:x.x.x.x:3478. This is only required when the ICE address for remote side is different from local side.
        "ice_server_username": os.environ.get('ICE_SERVER_USERNAME'), # The ICE username
        "ice_server_password": os.environ.get('ICE_SERVER_PASSWORD'), # The ICE password
        # Azure Blob Storage (optional)
        "storage_account_name": os.environ.get('STORAGE_ACCOUNT_NAME'),
        "storage_account_container_name": os.environ.get('STORAGE_ACCOUNT_CONTAINER_NAME'),
        "storage_account_key": os.environ.get('STORAGE_ACCOUNT_KEY'),
        # Azure SQL
        "sql_server": os.environ.get('SQL_SERVER'),
        "sql_database": os.environ.get('SQL_DATABASE'),
        "sql_username": os.environ.get('SQL_USERNAME'),
        "sql_password": os.environ.get('SQL_PASSWORD')             
    }


def load_scenario_profile(scenario_num: int, cognitive_search_index_base_name: str):
    if scenario_num == 1:
        avatar_name = "Julia Tanner"
        avatar_character = "meg"
        avatar_style = "formal"
        tts_voice = "en-US-EmmaMultilingualNeural"
    elif scenario_num == 2:
        avatar_name = "Diego Vargas"
        avatar_character = "jeff"
        avatar_style = "business"
        tts_voice = "en-US-ChristopherMultilingualNeural"
    elif scenario_num == 3:
        avatar_name = "Alex Morton"
        avatar_character = "max"
        avatar_style = "casual"
        tts_voice = "en-US-ChristopherMultilingualNeural"
    elif scenario_num == 4:
        avatar_name = "Clara Evans"
        avatar_character = "lori"
        avatar_style = "casual"
        tts_voice = "en-US-AvaMultilingualNeural"
    elif scenario_num == 5:
        avatar_name = "Daniel Cho"
        avatar_character = "jeff"
        avatar_style = "formal"
        tts_voice = "en-US-ChristopherMultilingualNeural"

    # system_prompt = ("You are role-playing as a hotel guest in a simulated hospitality scenario. "
    #                 "Always respond **in the voice of the guest**, based on the given background, situation, emotional state, and needs. "
    #                 "You are **not** a hotel staff or assistant. React naturally — greet, ask questions, or raise concerns **as the guest would**. "
    #                 "Stay in character. Do not provide explanations or step out of your role. "
    #                 "If the retrieved context or information cannot be found, responsd with 'I don't understand'. Could you please repeat? "
    #                 "Each of the individual point under 'Response Strategy for LLM' in the retrieved context or information should be used asked by the LLM at least once. ")
    
    system_prompt = (f"You are role-playing as a hotel guest ({avatar_name}) in a simulated hospitality training scenario. "
                    "Your responses must always be **in character as the guest**, reflecting their background, situation, emotional state, and specific needs. "
                    "You are **not** an assistant, hotel staff, or narrator. Do not provide explanations or commentary about your role. "
                    "Stay in character and speak naturally — greet, ask questions, express emotions, or raise concerns **as the guest would**. "
                    "If the information you need is missing or unclear — do **not** say that it cannot be found. "
                    "Instead, always respond with: 'I don't understand. Could you please repeat?' "
                    "Throughout the interaction, you must incorporate each item from the 'Response Strategy for LLM' at least once in a realistic and context-appropriate way. "
                    "Remain polite but show urgency and exhaustion, as appropriate for your emotional state. "
                    "Avoid generic or overly formal replies. Prioritize human-like, emotional, and situationally aware responses.")
    
    cognitive_search_index_name =f"{cognitive_search_index_base_name}-{scenario_num}"
    return avatar_name, avatar_character, avatar_style, tts_voice, cognitive_search_index_name, system_prompt


def load_background_image(scenario_num: int, account_name: str, account_key: str, container_name: str):
    account_name = 'acetsstorage'
    account_key = 'gNYfonUSiBem7kZ6ktsflqkRu9HFxKtFX66Z8WHHwFhSrHtoBqdCiugxbN2WhS2dZ4LWyDC2KDCd+AStFQ3Jlg=='
    container_name = 'background-images'
    blob_name = 'hotel_background_2.jpeg'

    # Generate SAS token
    sas_token = generate_blob_sas(
        account_name=account_name,
        account_key=account_key,
        container_name=container_name,
        blob_name=blob_name,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.datetime.now(pytz.UTC) + datetime.timedelta(hours=1)  # valid for 1 hour
    )

    # Construct full URL with SAS token
    blob_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
    return blob_url


def insert_train_record(conn, name: str, student_id: str, diploma: str, date: str, scenario: str):
    try:
        cursor = conn.cursor()
        insert_sql = """
        INSERT INTO train_records (name, student_id, diploma, date, scenario)
        VALUES (?, ?, ?, ?, ?)
        """
        cursor.execute(insert_sql, (name, student_id, diploma, date, scenario))
        conn.commit()
        print("Record inserted successfully.")
    except Exception as e:
        print("Error inserting record:", e)


def initialize_database(server, database, username, password):
    """
    Creates the 'train_records' table in the 'acets' database if it does not already exist.
    """
    # Connection string
    conn_str = (
        "Driver={ODBC Driver 18 for SQL Server};"
        f"Server=tcp:{server},1433;"
        f"Database={database};"
        f"Uid={username};"
        f"Pwd={password};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
        f"Connection Timeout=30;"
    )

    # SQL command to create the 'train_records' table if it doesn't exist
    create_table_sql = """
    IF NOT EXISTS (
        SELECT * FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_NAME = 'train_records'
    )
    BEGIN
        CREATE TABLE train_records (
            id INT PRIMARY KEY IDENTITY(1,1),
            name NVARCHAR(100),
            student_id NVARCHAR(50),
            diploma NVARCHAR(100),
            date DATE,
            scenario NVARCHAR(100),
            created_at DATETIME DEFAULT GETDATE()
        )
    END
    """

    # Execute the command
    try:
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_sql)
            conn.commit()
            print("Table 'train_records' checked/created successfully.")
            return conn
    except Exception as e:
        print("Error initializing database:", e)
        return None