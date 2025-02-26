# 使用 SageMaker PyTorch 基础镜像
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.0-gpu-py310

# 安装额外的依赖
RUN pip install "git+https://github.com/hairozen/anomalib.git@ai-vad-inference-improvements" 