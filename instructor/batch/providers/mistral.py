"""
OpenAI-specific batch processing implementation.

This module contains the OpenAI batch processing provider class.
"""

from typing import Any, Optional
from .base import BatchProvider
from ..models import BatchJobInfo


class MistralProvider(BatchProvider):
    """Mistral batch processing provider"""

    def submit_batch(
        self, file_path: str, metadata: Optional[dict[str, Any]] = None, **kwargs
    ) -> str:
        """Submit Mistral batch job"""
        try:
            import os
            from mistralai import Mistral

            if os.environ.get("MISTRAL_API_KEY"):
                client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
            else:
                raise ValueError(
                    "MISTRAL_API_KEY is not set. "
                    "Set it with `export MISTRAL_API_KEY=<your-api-key>`."
                )

            if metadata is None:
                metadata = {"description": "Instructor batch job"}

            with open(file_path, "rb") as f:
                batch_file = client.files.upload(
                    file={
                        "file_name": "batchinput.jsonl",
                        "content": f
                    },
                    purpose="batch"
                )

            ## Check if provider specific arguments are present.
            if not "model_name" in kwargs:
                raise Exception(f"Model for Mistral batch is missing")

            batch_job = client.batch.jobs.create(
                input_files=[batch_file.id],
                model=kwargs.get("model_name"),
                endpoint=kwargs.get("endpoint", "/v1/chat/completions"),
                timeout_hours=kwargs.get("completion_window", "24"),
                metadata=metadata,
            )
            return batch_job.id
        except Exception as e:
            raise Exception(f"Failed to submit Mistral batch: {e}") from e

    def get_status(self, batch_id: str) -> dict[str, Any]:
        """Get Mistral batch status"""
        try:
            import os
            from mistralai import Mistral

            if os.environ.get("MISTRAL_API_KEY"):
                client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
            else:
                raise ValueError(
                    "MISTRAL_API_KEY is not set. "
                    "Set it with `export MISTRAL_API_KEY=<your-api-key>`."
                )
            
            batch = client.batch.jobs.get(job_id=batch_id)
            return {
                "id": batch.id,
                "status": batch.status,
                "created_at": batch.created_at,
                "request_counts": {
                    "total": getattr(batch, "total_requests", 0),
                    "completed": getattr(batch, "sacceeded_requests", 0),
                    "failed": getattr(batch, "failed_requests", 0),
                },
            }
        except Exception as e:
            raise Exception(f"Failed to get Mistral batch status: {e}") from e

    def retrieve_results(self, batch_id: str) -> str:
        """Retrieve Mistral batch results"""
        try:
            import os
            from mistralai import Mistral

            if os.environ.get("MISTRAL_API_KEY"):
                client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
            else:
                raise ValueError(
                    "MISTRAL_API_KEY is not set. "
                    "Set it with `export MISTRAL_API_KEY=<your-api-key>`."
                )
            
            batch = client.batch.jobs.get(job_id=batch_id)

            if batch.status != "SUCCESS":
                raise Exception(f"Batch not completed, status: {batch.status}")

            if not batch.output_file:
                raise Exception("No output file available")

            file_response_stream = client.files.download(file_id=batch.output_file)

            file_response_stream.read()
            return file_response_stream.text
            
        except Exception as e:
            raise Exception(f"Failed to retrieve Mistral results: {e}") from e

    def download_results(self, batch_id: str, file_path: str) -> None:
        """Download Mistral batch results to a file"""
        try:
            import os
            from mistralai import Mistral

            if os.environ.get("MISTRAL_API_KEY"):
                client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
            else:
                raise ValueError(
                    "MISTRAL_API_KEY is not set. "
                    "Set it with `export MISTRAL_API_KEY=<your-api-key>`."
                )
            
            batch = client.batch.jobs.get(job_id=batch_id)

            if batch.status != "SUCCESS":
                raise Exception(f"Batch not completed, status: {batch.status}")

            if not batch.output_file:
                raise Exception("No output file available")

            file_response_stream = client.files.download(file_id=batch.output_file)

            with open(file_path, "w") as f:
                f.write(file_response_stream.read())
        except Exception as e:
            raise Exception(f"Failed to download Mistral results: {e}") from e

    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        """Cancel OpenAI batch job"""
        try:
            import os
            from mistralai import Mistral

            if os.environ.get("MISTRAL_API_KEY"):
                client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
            else:
                raise ValueError(
                    "MISTRAL_API_KEY is not set. "
                    "Set it with `export MISTRAL_API_KEY=<your-api-key>`."
                )
            
            batch = client.batch.jobs.cancel(job_id=batch_id)
            return batch.serialize_model()
        except Exception as e:
            raise Exception(f"Failed to cancel Mistral batch: {e}") from e

    def delete_batch(self, batch_id: str) -> dict[str, Any]:
        """Delete OpenAI batch job"""
        try:
            import os
            from mistralai import Mistral

            if os.environ.get("MISTRAL_API_KEY"):
                client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
            else:
                raise ValueError(
                    "MISTRAL_API_KEY is not set. "
                    "Set it with `export MISTRAL_API_KEY=<your-api-key>`."
                )
            
            # Mistral doesn't have a delete endpoint, so we'll return the batch info
            batch = client.batch.jobs.get(job_id=batch_id)
            return {
                "id": batch.id,
                "status": batch.status,
                "message": "Mistral does not support batch deletion",
            }
        except Exception as e:
            raise Exception(f"Failed to delete Mistral batch: {e}") from e

    def list_batches(self, limit: int = 10) -> list[BatchJobInfo]:
        """List OpenAI batch jobs"""
        try:
            import os
            from mistral import Mistral

            if os.environ.get("MISTRAL_API_KEY"):
                client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
            else:
                raise ValueError(
                    "MISTRAL_API_KEY is not set. "
                    "Set it with `export MISTRAL_API_KEY=<your-api-key>`."
                )
            
            batches = client.batch.jobs.list(page_size=limit)
            return [
                BatchJobInfo.from_mistral(batch.serialize_model()) for batch in batches.data
            ]
        except Exception as e:
            raise Exception(f"Failed to list Mistral batches: {e}") from e
