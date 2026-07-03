import os

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name=os.getenv("AWS_REGION")
)

BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")


def upload_file(local_path, s3_key):
    try:
        s3.upload_file(local_path, BUCKET_NAME, s3_key)
        print(f"✅ Uploaded: {s3_key}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")


def key_exists(s3_key: str) -> bool:
    """head_object check; returns False only on a confirmed 404, raises on anything else."""
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=s3_key)
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return False
        raise


def upload_file_append_only(local_path, s3_key):
    """Append-only upload: raises if `s3_key` already exists instead of overwriting,
    and raises (rather than swallowing) any credential/network/client error -- callers
    that need retry-without-overwrite guarantees must not catch exceptions from this."""
    if key_exists(s3_key):
        raise FileExistsError(
            f"S3 key 's3://{BUCKET_NAME}/{s3_key}' already exists -- this upload path is "
            f"append-only and must never overwrite a previously written batch. If state still "
            f"points at this batch, a prior run likely uploaded successfully but failed during "
            f"local finalization afterward -- inspect/delete this S3 key (or any leftover local "
            f".tmp file for this batch) before retrying."
        )
    s3.upload_file(local_path, BUCKET_NAME, s3_key)
    print(f"✅ Uploaded (append-only): s3://{BUCKET_NAME}/{s3_key}")
