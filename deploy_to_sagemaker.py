import boto3
import sagemaker
from datetime import datetime
import os
import json

def deploy_model_to_sagemaker(
    model_artifacts_dir,
    aws_region,
    aws_account,
    instance_type="ml.g5.xlarge",
    initial_instance_count=1
):
    """
    将模型部署到 SageMaker
    
    参数:
        model_artifacts_dir: 包含模型文件的目录
        aws_region: AWS 区域
        aws_account: AWS 账号ID
        instance_type: 实例类型
        initial_instance_count: 初始实例数量
    """
    
    # 获取当前时间戳
    current_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    # 创建 S3 客户端
    s3 = boto3.resource('s3')
    
    # 创建 SageMaker 客户端
    sagemaker_client = boto3.client('sagemaker')
    
    # 获取 SageMaker 执行角色
    role = sagemaker.get_execution_role()
    
    # 第1步: 打包模型文件
    print("正在打包模型文件...")
    model_files = [
        'ai_vad_weights.pth',
        'ai_vad_banks.joblib',
        'ai_vad_config.yaml',
        'inference.py'
    ]
    
    os.system(f"tar -czvf model.tar.gz -C {model_artifacts_dir} {' '.join(model_files)}")
    
    # 第2步: 上传模型到 S3
    print("正在上传模型到 S3...")
    bucket_name = "ai-vad"
    model_s3_key = f"{current_datetime}/model.tar.gz"
    
    s3.meta.client.upload_file(
        "model.tar.gz", 
        bucket_name,
        model_s3_key
    )
    
    model_data_url = f"s3://{bucket_name}/{model_s3_key}"
    
    # 第3步: 创建 SageMaker 模型
    print("正在创建 SageMaker 模型...")
    model_name = f"ai-vad-model-{current_datetime}"
    
    primary_container = {
        "Image": f"{aws_account}.dkr.ecr.{aws_region}.amazonaws.com/ai-vad-image:latest",
        "ModelDataUrl": model_data_url
    }
    
    create_model_response = sagemaker_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer=primary_container
    )
    
    # 第4步: 创建端点配置
    print("正在创建端点配置...")
    endpoint_config_name = f"ai-vad-config-{current_datetime}"
    
    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            "InstanceType": instance_type,
            "InitialVariantWeight": 1,
            "InitialInstanceCount": initial_instance_count,
            "ModelName": model_name,
            "VariantName": "AllTraffic"
        }]
    )
    
    # 第5步: 创建端点
    print("正在创建端点...")
    endpoint_name = f"ai-vad-endpoint-{current_datetime}"
    
    sagemaker_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )
    
    print("正在等待端点部署完成...")
    waiter = sagemaker_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)
    
    print(f"端点部署完成! 端点名称: {endpoint_name}")
    return endpoint_name

def test_endpoint(endpoint_name, image_path):
    """
    测试已部署的端点
    
    参数:
        endpoint_name: 端点名称
        image_path: 测试图片路径
    """
    print("正在测试端点...")
    
    # 读取测试图片
    with open(image_path, "rb") as f:
        payload = f.read()
    
    # 创建 SageMaker Runtime 客户端
    runtime = boto3.client("runtime.sagemaker")
    
    # 调用端点
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="image/x-image",
        Body=payload
    )
    
    # 解析响应
    result = json.loads(response["Body"].read().decode())
    print("预测结果:", result)
    return result

if __name__ == "__main__":
    # 配置参数
    MODEL_DIR = "./model_artifacts"  # 模型文件目录
    AWS_REGION = "us-east-1"         # AWS 区域
    AWS_ACCOUNT = "your_account_id"  # AWS 账号ID
    
    # 部署模型
    endpoint_name = deploy_model_to_sagemaker(
        model_artifacts_dir=MODEL_DIR,
        aws_region=AWS_REGION,
        aws_account=AWS_ACCOUNT
    )
    
    # 测试端点
    test_image = "./test_images/test.jpg"
    test_endpoint(endpoint_name, test_image) 