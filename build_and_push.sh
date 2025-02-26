#!/bin/bash

# 设置变量
REGION="us-east-1"
ACCOUNT="your_account_id"

# 登录到 ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com

# 创建 ECR 仓库
aws ecr create-repository --repository-name ai-vad-image

# 构建镜像
docker build -t ai-vad-image .

# 标记镜像
docker tag ai-vad-image:latest $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/ai-vad-image:latest

# 推送镜像到 ECR
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/ai-vad-image:latest 