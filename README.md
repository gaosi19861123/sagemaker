# AI VAD 模型 SageMaker 部署指南

本项目提供了将 AI VAD (视频异常检测) 模型部署到 AWS SageMaker 的完整流程。该模型基于论文"基于属性的表示，用于准确且可解释的视频异常检测"实现。

## 项目概述

AI VAD 是一个视频异常检测模型，可以检测视频中的异常行为。该项目提供了将模型部署到 AWS SageMaker 进行推理的完整解决方案。

## 主要特性

- 支持单帧图像和批量视频帧的异常检测
- 提供 REST API 接口进行在线推理
- 支持从 S3 读取输入数据
- 可配置的模型参数和推理设置
- GPU 加速支持

## 系统要求

- AWS 账号和相关配置
- Docker
- Python 3.10+
- AWS CLI
- 足够的 GPU 资源（推荐使用 ml.g5.xlarge 或更高配置）

## 项目结构

```
.
├── README.md                 # 项目文档
├── Dockerfile               # Docker 镜像构建文件
├── build_and_push.sh       # ECR 镜像构建和推送脚本
├── deploy_to_sagemaker.py  # SageMaker 部署脚本
├── inference.py            # 模型推理代码
└── model_artifacts/        # 模型文件目录
    ├── ai_vad_weights.pth  # 模型权重
    ├── ai_vad_banks.joblib # 模型记忆库
    └── ai_vad_config.yaml  # 模型配置文件
```

## 安装步骤

1. 克隆项目代码：
```bash
git clone <repository_url>
cd ai-vad-sagemaker
```

2. 配置 AWS 凭证：
```bash
aws configure
```

3. 修改配置文件：
   - 在 `build_and_push.sh` 中设置正确的 AWS Region 和 Account ID
   - 在 `deploy_to_sagemaker.py` 中设置相应的配置参数

4. 构建并推送 Docker 镜像：
```bash
chmod +x build_and_push.sh
./build_and_push.sh
```

## 部署流程

1. 准备模型文件：
   - 将模型权重文件 `ai_vad_weights.pth` 放入 `model_artifacts` 目录
   - 将模型记忆库文件 `ai_vad_banks.joblib` 放入 `model_artifacts` 目录
   - 将模型配置文件 `ai_vad_config.yaml` 放入 `model_artifacts` 目录

2. 运行部署脚本：
```bash
python deploy_to_sagemaker.py
```

3. 等待部署完成，获取端点名称。

## API 使用说明

模型支持两种输入格式：

1. 单帧图像：
```python
# Content-Type: application/x-image 或 image/x-image
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="image/x-image",
    Body=image_bytes
)
```

2. 批量图像（从 S3）：
```python
# Content-Type: application/json
payload = {
    "images": ["s3://bucket-name/path/to/image1.jpg", "s3://bucket-name/path/to/image2.jpg"],
    "clip": false  # 设置为 true 启用视频片段模式
}
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps(payload)
)
```

响应格式：
```json
{
    "pred_boxes": [[x1, y1, x2, y2], ...],  # 检测到的异常区域边界框
    "box_scores": [score1, score2, ...],     # 每个边界框的异常分数
    "pred_scores": [frame_score1, ...]       # 每帧的整体异常分数
}
```

## 配置说明

主要配置参数（在 `ai_vad_config.yaml` 中设置）：

- `box_score_thresh`: 边界框分数阈值
- `persons_only`: 是否只检测人物
- `min_bbox_area`: 最小边界框面积
- `max_bbox_overlap`: 最大边界框重叠度
- `use_velocity_features`: 是否使用速度特征
- `use_pose_features`: 是否使用姿态特征
- `use_deep_features`: 是否使用深度特征

## 性能优化

- 使用 GPU 实例（推荐 ml.g5.xlarge 或更高配置）
- 适当调整批处理大小
- 根据需求配置实例数量和自动扩缩容策略

## 故障排除

1. 如果遇到内存不足错误，尝试：
   - 减小批处理大小
   - 使用更大内存的实例类型
   
2. 如果遇到超时错误，尝试：
   - 增加端点超时设置
   - 减小输入数据大小

3. 如果遇到模型加载错误，检查：
   - 模型文件是否完整
   - 配置文件参数是否正确

## 许可证

[添加许可证信息]

## 联系方式

[添加联系方式] 