import os
import uuid
import base64
import numpy as np
import pandas as pd
from glob import glob
from azure.cosmos import CosmosClient, PartitionKey, exceptions

# Set variables
include_vector_indexing = False

# Replace with your actual connection string
COSMOS_CONNECTION_STRING = "AccountEndpoint=https://acets-dev-cosmos-db.documents.azure.com:443/;AccountKey=rozcRi8nGjbJ7fyTlNPNyaz1P5cDkk1XyRdd3PYp1r7QYyYUrhBignPcQqnNrcK3rnIUadeof7wCACDbTvjRBQ==;"

# Initialize Cosmos client
client = CosmosClient.from_connection_string(COSMOS_CONNECTION_STRING)

# Define database and container names
DATABASE_NAME = "acets_project_database"
CONTAINER_NAME = "scenarios"
PARTITION_KEY_PATH = "/dataset_type"  # Use dataset_type as partition key

# === Step 1: Delete the Existing Database (Optional) ===
try:
    client.delete_database(DATABASE_NAME)
    print(f"Deleted existing database: {DATABASE_NAME}")
except exceptions.CosmosResourceNotFoundError:
    print(f"Database '{DATABASE_NAME}' does not exist, skipping deletion.")

# === Step 2: Create a New Database ===
database = client.create_database(DATABASE_NAME)
print(f"Database '{DATABASE_NAME}' has been created.")

# === Step 3: Create a Single Container ===
container = database.create_container(
    id=CONTAINER_NAME,
    partition_key=PartitionKey(path=PARTITION_KEY_PATH)  # Using dataset_type as partition key
)
print(f"Created container: {CONTAINER_NAME}")


def clean_document(doc):
    """ Cleans the document by replacing NaN values and ensuring proper formatting. """
    cleaned_doc = {}
    for key, value in doc.items():
        if isinstance(value, float) and np.isnan(value):  # Replace NaN values with None
            cleaned_doc[key] = None
        elif isinstance(value, str) and value.lower() == "nil":
            cleaned_doc[key] = None
        elif isinstance(value, str):
            cleaned_doc[key] = value.replace("\x95", "").strip()  # Remove special characters
        else:
            cleaned_doc[key] = value
    return cleaned_doc


# Function to encode the _rid in base64
def encode_base64(value: str) -> str:
    """Encodes the given string in base64."""
    if value:
        return base64.b64encode(value.encode('utf-8')).decode('utf-8')
    return value


# Function to clean and update _rid after data is inserted
def encode_rid_and_save(client, database_name):
    # Connect to the database and container
    database = client.get_database_client(database_name)

    # Iterate through all containers
    for container in database.list_containers():
        container_name = container["id"]
        print(f"Processing container: {container_name}")

        # Query for documents to fetch all items in the container
        query = "SELECT * FROM c"
        items = list(database.get_container_client(container_name).query_items(query, enable_cross_partition_query=True))

        # Update the _rid field by encoding it in base64 and re-insert the documents
        for item in items:
            print(f"Found document with _rid: {item['_rid']}")
            
            # Encode _rid to base64
            encoded_rid = encode_base64(item["_rid"])

            # Update the document with the new encoded _rid
            item["_rid"] = encoded_rid

            # Replace the document in Cosmos DB
            database.get_container_client(container_name).upsert_item(item)
            print(f"Updated document with new encoded _rid: {encoded_rid}")


# Obtain the full CSV file list
cwd = os.getcwd()
csv_list = glob(os.path.join(cwd, "full_scenarios.csv"))
print(f"Found {len(csv_list)} CSV files to process.")


# === Step 4: Process Each CSV File and Insert Data ===
for single_csv_filepath in csv_list:
    dataset_type = os.path.basename(single_csv_filepath).replace(".csv", "")  # Dataset identifier

    # Read the CSV file
    df = pd.read_csv(single_csv_filepath, encoding="latin1")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Drop unwanted columns

    # Insert each row into Cosmos DB
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx == 1:
            item = row.to_dict()
            cleaned_item = clean_document(item)
            cleaned_item["id"] = str(item.get("id", uuid.uuid4()))
            # cleaned_item["dataset_type"] = dataset_type  # Add dataset type field
            # print(f'cleaned_item: {cleaned_item}')
            # print("=====")

            # Insert into Cosmos DB
            container.create_item(body=cleaned_item)
            print(f"Inserted document into '{CONTAINER_NAME}' with dataset_type: {dataset_type}")


# After data is inserted, encode _rid for all documents
encode_rid_and_save(client, DATABASE_NAME)