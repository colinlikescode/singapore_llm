import boto3
import logging
from urllib.parse import urlparse

def download_file(s3_url, local_path, region="us-east-1"):
    parsed = urlparse(s3_url)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    s3 = boto3.client('s3', region_name=region)
    try:
        with open(local_path, "wb") as f:
            s3.download_fileobj(bucket, key, f)
        logging.info(f"File downloaded from {s3_url} to {local_path}")
    except Exception as e:
        logging.error(f"Failed to download file from {s3_url}: {e}")
        raise
