from google.cloud import storage
from google.oauth2 import service_account
from googleapiclient import discovery
import os 

# Instance info abstracted-away on Kubernetes. 
def assign_default(var, default='NOT_DEFINED'):
    if var not in os.environ:
        os.environ[var] = default 
    pass
assign_default('ZONE') 
assign_default('PROJECT') 
assign_default('INSTANCE') 

def download_blob(source_blob_name, destination_file_name, bucket_name=os.environ['BUCKET_NAME']):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"
    storage_client = storage.Client.from_service_account_json('/app/service-account.json')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )
    pass

def upload_blob(source_file_name, destination_blob_name, bucket_name=os.environ['BUCKET_NAME']):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"
    storage_client = storage.Client.from_service_account_json('/app/service-account.json')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )
    pass

def shutdown(zone=os.environ['ZONE'], project=os.environ['PROJECT'], instance=os.environ['INSTANCE']):
    credentials = service_account.Credentials.from_service_account_file('/app/service-account.json')
    service = discovery.build('compute', 'v1', credentials=credentials)
    request = service.instances().stop(project=project, zone=zone, instance=instance)
    print('Shutting down...')
    response = request.execute()
    pass

def delete_cluster(zone=os.environ['ZONE'], project=os.environ['PROJECT'], cluster=os.environ['INSTANCE']): 
    credentials = service_account.Credentials.from_service_account_file('/app/service-account.json') 
    service = discovery.build('container', 'v1', credentials=credentials) 
    request = service.projects().zones().clusters().delete(projectId=project, zone=zone, clusterId=cluster) 
    print('Deleting cluster...') 
    response = request.execute() 
    pass 
