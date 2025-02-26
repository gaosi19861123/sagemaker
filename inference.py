import os
import json
import joblib
import torch
from PIL import Image
import numpy as np
import io
import boto3
from enum import Enum
from urllib.parse import urlsplit
from omegaconf import OmegaConf
from anomalib.data.utils import read_image, InputNormalizationMethod, get_transforms
from anomalib.models.ai_vad.torch_model import AiVadModel

device = "cuda"

class PredictMode(Enum):
    frame = 1
    batch = 2
    clip = 3

def model_fn(model_dir):
    """加载模型,只在启动时调用一次"""
    # 加载配置文件
    config = OmegaConf.load(os.path.join(model_dir, "ai_vad_config.yaml"))
    config_model = config.model

    # 加载模型
    model = AiVadModel(
        box_score_thresh=config_model.box_score_thresh,
        persons_only=config_model.persons_only,
        min_bbox_area=config_model.min_bbox_area,
        max_bbox_overlap=config_model.max_bbox_overlap,
        enable_foreground_detections=config_model.enable_foreground_detections,
        foreground_kernel_size=config_model.foreground_kernel_size,
        foreground_binary_threshold=config_model.foreground_binary_threshold,
        n_velocity_bins=config_model.n_velocity_bins,
        use_velocity_features=config_model.use_velocity_features,
        use_pose_features=config_model.use_pose_features,
        use_deep_features=config_model.use_deep_features,
        n_components_velocity=config_model.n_components_velocity,
        n_neighbors_pose=config_model.n_neighbors_pose,
        n_neighbors_deep=config_model.n_neighbors_deep,
    )

    # 加载模型权重
    model.load_state_dict(torch.load(os.path.join(model_dir, "ai_vad_weights.pth"), map_location=device), strict=False)

    # 加载记忆库
    velocity_estimator_memory_bank, pose_estimator_memory_bank, appearance_estimator_memory_bank = joblib.load(os.path.join(model_dir, "ai_vad_banks.joblib"))
    if velocity_estimator_memory_bank is not None:
        model.density_estimator.velocity_estimator.memory_bank = velocity_estimator_memory_bank
    if pose_estimator_memory_bank is not None:
        model.density_estimator.pose_estimator.memory_bank = pose_estimator_memory_bank
    if appearance_estimator_memory_bank is not None:
        model.density_estimator.appearance_estimator.memory_bank = appearance_estimator_memory_bank
    model.density_estimator.fit()

    # 将模型移至设备
    model = model.to(device)

    # 获取转换
    transform_config = config.dataset.transform_config.eval if "transform_config" in config.dataset.keys() else None
    image_size = (config.dataset.image_size[0], config.dataset.image_size[1])
    center_crop = config.dataset.get("center_crop")
    center_crop = tuple(center_crop) if center_crop is not None else None
    normalization = InputNormalizationMethod(config.dataset.normalization)
    transform = get_transforms(config=transform_config, image_size=image_size, center_crop=center_crop, normalization=normalization)

    return model, transform

def input_fn(request_body, request_content_type):
    """处理输入数据"""
    print("input_fn-----------------------")

    if request_content_type in ("application/x-image", "image/x-image"):
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        numpy_array = np.array(image)
        print("numpy_array.shape", numpy_array.shape)
        print("input_fn-----------------------")
        return [numpy_array], PredictMode.frame

    elif request_content_type == "application/json":
        request_body_json = json.loads(request_body)
        s3_uris = request_body_json.get("images", [])

        if len(s3_uris) == 0:
            raise ValueError("Images is a required key and should contain at least a list of one S3 URI")

        s3 = boto3.client("s3")
        frame_paths = []
        for s3_uri in s3_uris:
            parsed_url = urlsplit(s3_uri)
            bucket_name = parsed_url.netloc
            object_key = parsed_url.path.lstrip('/')
            local_frame_path = f"/tmp/{s3_uri.replace('/', '_')}"
            s3.download_file(bucket_name, object_key, local_frame_path)
            frame_paths.append(local_frame_path)

        frames = np.stack([torch.Tensor(read_image(frame_path)) for frame_path in frame_paths], axis=0)
        predict_mode = PredictMode.clip if request_body_json.get("clip", False) else PredictMode.batch

        print("frames.shape", frames.shape)
        print("predict_mode", predict_mode)
        print("input_fn-----------------------")

        return frames, predict_mode

    raise ValueError(f"Content type {request_content_type} is not supported")

def predict_fn(input_data, model):
    """执行模型推理"""
    print("predict_fn-----------------------")

    model, transform = model
    frames, predict_mode = input_data

    processed_data = {}
    processed_data["image"] = [transform(image=frame)["image"] for frame in frames]
    processed_data["image"] = torch.stack(processed_data["image"])

    image = processed_data["image"].to(device)

    if predict_mode == PredictMode.clip:
        image = image.unsqueeze(0)

    print("image.shape", image.shape)

    model.eval()
    with torch.no_grad():
        boxes, anomaly_scores, image_scores = model(image)

    print("boxes_len", [len(b) for b in boxes])

    processed_data["pred_boxes"] = [box.int() for box in boxes]
    processed_data["box_scores"] = [score.to(device) for score in anomaly_scores]
    processed_data["pred_scores"] = torch.Tensor(image_scores).to(device)

    print("predict_fn-----------------------")
    return processed_data

def output_fn(prediction, accept):
    """格式化输出"""
    print("output_fn-----------------------")

    if accept != "application/json":
        raise ValueError(f"Accept type {accept} is not supported")

    for key in prediction:
        if isinstance(prediction[key], torch.Tensor):
            prediction[key] = prediction[key].tolist()
        elif isinstance(prediction[key], list):
            prediction[key] = [tensor.tolist() if isinstance(tensor, torch.Tensor) else tensor for tensor in prediction[key]]

    print("output_fn-----------------------")
    return json.dumps(prediction), accept
