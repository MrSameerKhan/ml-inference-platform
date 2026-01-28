import argparse
import json
import boto3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="ap-south-1")
    ap.add_argument("--endpoint-name", required=True)
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    rt = boto3.client("sagemaker-runtime", region_name=args.region)

    payload = {"text": args.text}
    resp = rt.invoke_endpoint(
        EndpointName=args.endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    body = resp["Body"].read().decode("utf-8")
    print(body)


# python sagemaker\invoke.py --region ap-south-1 --endpoint-name textcls-endpoint-dev --text "refund my money"

if __name__ == "__main__":
    main()
