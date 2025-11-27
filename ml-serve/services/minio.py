from typing import TypeVar, Type, Optional
from pydantic import BaseModel

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import yaml

from core.config import settings

T = TypeVar("T", bound=BaseModel)


class MinioService:
    def __init__(self) -> None:
        self.session = boto3.session.Session()
        self.s3_client = None

    def __enter__(self) -> "MinioService":
        self.s3_client = self.session.client(
            service_name="s3",
            endpoint_url=settings.MINIO_ENDPOINT_URL,
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 10, "mode": "standard"},
            ),
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.s3_client:
            self.s3_client.close()

    @staticmethod
    def load_and_validate_yaml(
        s3_client,
        bucket: str,
        key: str,
        model: Type[T],
        version_id: Optional[str] = None,
    ) -> T:
        kwargs = {"Bucket": bucket, "Key": key}
        if version_id is not None:
            kwargs["VersionId"] = version_id

        resp = s3_client.get_object(**kwargs)
        raw_bytes: bytes = resp["Body"].read()

        text = raw_bytes.decode("utf-8")
        data = yaml.safe_load(text)

        try:
            return model.model_validate(data)
        except AttributeError:
            return model.parse_obj(data)

    def _load_latest(
        self,
        key: str,
        model: Type[T],
    ) -> T:
        return self.load_and_validate_yaml(
            s3_client=self.s3_client,
            bucket=settings.MINIO_BUCKET_NAME,
            key=key,
            model=model,
        )

    def _load_latest_stable(
        self,
        key: str,
        model: Type[T],
    ) -> T:
        """
        Ищет среди версий объекта последнюю, помеченную тегом stable=true,
        и валидирует её.

        Ожидается, что ты выставляешь у версии теги:
            Key = stable, Value = true
        """
        try:
            resp = self.s3_client.list_object_versions(
                Bucket=settings.MINIO_BUCKET_NAME,
                Prefix=key,
            )
        except ClientError as e:
            raise RuntimeError(f"Failed to list versions for '{key}'") from e

        versions = resp.get("Versions", [])
        if not versions:
            raise RuntimeError(f"No versions found for '{key}'")

        versions_sorted = sorted(
            [v for v in versions if v.get("Key") == key],
            key=lambda v: v["LastModified"],
            reverse=True,
        )

        for v in versions_sorted:
            version_id = v["VersionId"]
            try:
                tags_resp = self.s3_client.get_object_tagging(
                    Bucket=settings.MINIO_BUCKET_NAME,
                    Key=key,
                    VersionId=version_id,
                )
            except ClientError:
                continue

            tagset = tags_resp.get("TagSet", [])
            is_stable = any(
                t.get("Key") == "stable" and str(t.get("Value")).lower() == "true"
                for t in tagset
            )
            if not is_stable:
                continue

            return self.load_and_validate_yaml(
                s3_client=self.s3_client,
                bucket=settings.MINIO_BUCKET_NAME,
                key=key,
                model=model,
                version_id=version_id,
            )

        raise RuntimeError(f"No stable version found for '{key}'")

    async def get_config(
        self,
        object_name: str,
        model: Type[T],
        fallback_object_name: Optional[str] = None,
    ) -> T:
        """
        1) Пытается взять **последнюю** версию object_name.
        2) Если не вышло — пытается взять **последнюю stable-версию**:
           - по fallback_object_name, если он указан
           - иначе по тому же object_name.

        stable определяется тегом версии: stable=true.
        """
        errors: list[str] = []

        try:
            return self._load_latest(
                key=object_name,
                model=model,
            )
        except Exception as e: 
            errors.append(f"current(latest) '{object_name}' error: {repr(e)}")

        print (fallback_object_name, object_name)

        stable_key = fallback_object_name or object_name
        try:
            return self._load_latest_stable(
                key=stable_key,
                model=model,
            )
        except Exception as e:  
            errors.append(f"stable(latest) '{stable_key}' error: {repr(e)}")

        raise RuntimeError("Failed to load configs: " + " | ".join(errors))




'''
from fastapi import Depends, HTTPException

async def get_antifraud_config(request: Request) -> AntifraudConfig:
    cfg = request.app.state.antifraud_config
    if cfg is None:
        raise HTTPException(status_code=503, detail="Config not loaded yet")
    return cfg

'''