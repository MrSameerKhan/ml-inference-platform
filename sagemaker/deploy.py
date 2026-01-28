import argparse
import time
import boto3
from botocore.exceptions import ClientError


def ensure_model(sm, model_name: str, image_uri: str, model_data_url: str, execution_role_arn: str):
    try:
        sm.describe_model(ModelName=model_name)
        print(f"[OK] Model exists: {model_name}")
        return
    except ClientError as e:
        if e.response["Error"]["Code"] != "ValidationException":
            raise

    print(f"[CREATE] Model: {model_name}")
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
            # Optional env; your app uses default /opt/ml/model so not required.
            # "Environment": {"MODEL_DIR": "/opt/ml/model"},
        },
        ExecutionRoleArn=execution_role_arn,
    )


def ensure_endpoint_config(sm, endpoint_config_name: str, model_name: str, instance_type: str, initial_instance_count: int):
    try:
        sm.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        print(f"[OK] EndpointConfig exists: {endpoint_config_name}")
        return
    except ClientError as e:
        if e.response["Error"]["Code"] != "ValidationException":
            raise

    print(f"[CREATE] EndpointConfig: {endpoint_config_name}")
    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": initial_instance_count,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
    )


def create_or_update_endpoint(sm, endpoint_name: str, endpoint_config_name: str):
    try:
        desc = sm.describe_endpoint(EndpointName=endpoint_name)
        status = desc["EndpointStatus"]
        print(f"[INFO] Endpoint exists: {endpoint_name} (status={status})")
        print(f"[UPDATE] Updating endpoint to config: {endpoint_config_name}")
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
        return "updated"
    except ClientError as e:
        if e.response["Error"]["Code"] != "ValidationException":
            raise

    print(f"[CREATE] Endpoint: {endpoint_name}")
    sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
    return "created"


def wait_for_inservice(sm, endpoint_name: str, poll_seconds: int = 30, timeout_seconds: int = 3600):
    start = time.time()
    while True:
        desc = sm.describe_endpoint(EndpointName=endpoint_name)
        status = desc["EndpointStatus"]
        if status == "InService":
            print(f"[OK] Endpoint InService: {endpoint_name}")
            return
        if status in ("Failed", "OutOfService"):
            reason = desc.get("FailureReason", "unknown")
            raise RuntimeError(f"Endpoint status={status}. Reason: {reason}")
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for endpoint InService after {timeout_seconds}s. Current status={status}")
        print(f"[WAIT] status={status} ...")
        time.sleep(poll_seconds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="ap-south-1")
    ap.add_argument("--image-uri", required=True)
    ap.add_argument("--model-data-url", required=True)  # s3://.../saved_model.tar.gz
    ap.add_argument("--execution-role-arn", required=True)

    ap.add_argument("--model-name", required=True)
    ap.add_argument("--endpoint-config-name", required=True)
    ap.add_argument("--endpoint-name", required=True)

    ap.add_argument("--instance-type", default="ml.m5.large")
    ap.add_argument("--initial-instance-count", type=int, default=1)
    args = ap.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)

    ensure_model(
        sm,
        model_name=args.model_name,
        image_uri=args.image_uri,
        model_data_url=args.model_data_url,
        execution_role_arn=args.execution_role_arn,
    )

    ensure_endpoint_config(
        sm,
        endpoint_config_name=args.endpoint_config_name,
        model_name=args.model_name,
        instance_type=args.instance_type,
        initial_instance_count=args.initial_instance_count,
    )

    action = create_or_update_endpoint(sm, endpoint_name=args.endpoint_name, endpoint_config_name=args.endpoint_config_name)
    print(f"[INFO] Endpoint {action}: {args.endpoint_name}")

    wait_for_inservice(sm, endpoint_name=args.endpoint_name)

# python sagemaker\deploy.py `
#   --region ap-south-1 `
#   --image-uri 570617927874.dkr.ecr.ap-south-1.amazonaws.com/ml-infer:b2-91db72b `
#   --model-data-url s3://sameer-ml-artifacts-ap-south-1/ml/textcls/versioned/v3.0.0/saved_model.tar.gz `
#   --execution-role-arn arn:aws:iam::570617927874:role/sagemaker-textcls-exec-role `
#   --model-name textcls-model-v3-0-0-b2-91db72b `
#   --endpoint-config-name textcls-config-v3-0-0-b2-91db72b `
#   --endpoint-name textcls-endpoint-dev
if __name__ == "__main__":
    main()
