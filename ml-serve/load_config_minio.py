import argparse
from typing import Optional

import boto3
from botocore.config import Config
import yaml

from core.config import settings
from models.config import AntifraudConfig


def load_local_config(path: str) -> AntifraudConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    try:
        cfg = AntifraudConfig.model_validate(data)  
    except AttributeError:
        cfg = AntifraudConfig.parse_obj(data)     

    return cfg


def upload_to_minio(
    cfg: AntifraudConfig,
    object_key: str,
    mark_stable: bool = False,
) -> None:
    session = boto3.session.Session()
    s3_client = session.client(
        service_name="s3",
        endpoint_url=settings.MINIO_ENDPOINT_URL,
        aws_access_key_id=settings.MINIO_ACCESS_KEY,
        aws_secret_access_key=settings.MINIO_SECRET_KEY,
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 10, "mode": "standard"},
        ),
    )

    yaml_text = yaml.safe_dump(
        cfg.model_dump(mode="python"),
        sort_keys=False,
        allow_unicode=True,
    )
    body = yaml_text.encode("utf-8")

    resp = s3_client.put_object(
        Bucket=settings.MINIO_BUCKET_NAME,
        Key=object_key,
        Body=body,
    )

    version_id: Optional[str] = resp.get("VersionId")
    print(f"Uploaded config to s3://{settings.MINIO_BUCKET_NAME}/{object_key}")
    if version_id:
        print(f"New VersionId: {version_id}")

    if mark_stable and version_id:
        s3_client.put_object_tagging(
            Bucket=settings.MINIO_BUCKET_NAME,
            Key=object_key,
            VersionId=version_id,
            Tagging={
                "TagSet": [
                    {"Key": "stable", "Value": "true"},
                ]
            },
        )

def main() -> None:
    parser = argparse.ArgumentParser(description="Upload antifraud config to MinIO")
    parser.add_argument(
        "--file",
        "-f",
        required=True,
        help="Path to local YAML config file",
    )
    parser.add_argument(
        "--key",
        "-k",
        default="configs/antifraud_txn_v1.yaml",
        help="Object key in MinIO (default: configs/antifraud_txn_v1.yaml)",
    )
    parser.add_argument(
        "--stable",
        action="store_true",
        help="Mark uploaded version as stable=true",
    )

    args = parser.parse_args()

    cfg = load_local_config(args.file)

    upload_to_minio(
        cfg=cfg,
        object_key=args.key,
        mark_stable=args.stable,
    )


if __name__ == "__main__":
    main()
