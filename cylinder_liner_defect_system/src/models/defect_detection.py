"""
缺陷检测模型 - 基于多模态融合的深度学习模型
Defect Detection Model - Multi-modal fusion deep learning model
"""

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class MultiModalFusionNetwork(nn.Module):
    """多模态融合网络"""

    def __init__(self, num_defect_classes: int = 6, num_grade_classes: int = 4):
        super().__init__()

        self.num_defect_classes = num_defect_classes
        self.num_grade_classes = num_grade_classes

        # 2D图像特征提取器 (ResNet50 backbone)
        self.image_encoder = ImageEncoder()
        self.image_feature_dim = 2048

        # 3D点云特征提取器
        self.pointcloud_encoder = PointCloudEncoder()
        self.pointcloud_feature_dim = 1024

        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.image_feature_dim + self.pointcloud_feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # 多头注意力机制
        self.attention = MultiHeadAttention(512, num_heads=8)

        # 缺陷检测头
        self.detection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_defect_classes),
        )

        # 等级评估头
        self.grading_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_grade_classes),
        )

        # 分割头
        self.segmentation_head = SegmentationHead(512, num_defect_classes)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(
        self, image: torch.Tensor, pointcloud: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """前向传播"""
        # 提取2D特征
        image_features = self.image_encoder(image)

        # 提取3D特征
        pointcloud_features = self.pointcloud_encoder(pointcloud)

        # 特征融合
        fused_features = torch.cat([image_features, pointcloud_features], dim=1)
        fused_features = self.fusion_layer(fused_features)

        # 注意力增强
        attended_features = self.attention(fused_features)

        # 缺陷检测
        detection_output = self.detection_head(attended_features)

        # 等级评估
        grading_output = self.grading_head(attended_features)

        # 分割输出
        segmentation_output = self.segmentation_head(
            attended_features, image.shape[-2:]
        )

        return {
            'detection': detection_output,
            'grading': grading_output,
            'segmentation': segmentation_output,
            'features': attended_features,
        }


class ImageEncoder(nn.Module):
    """2D图像编码器"""

    def __init__(self):
        super().__init__()

        # 使用预训练的ResNet50作为backbone
        import torchvision.models as models

        resnet = models.resnet50(pretrained=True)

        # 移除最后的全连接层
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # 添加自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 特征维度调整
        self.feature_proj = nn.Linear(2048, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 特征提取
        features = self.backbone(x)

        # 全局平均池化
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)

        # 特征投影
        features = self.feature_proj(features)

        return features


class PointCloudEncoder(nn.Module):
    """3D点云编码器"""

    def __init__(self):
        super().__init__()

        # PointNet架构
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # 全局特征提取
        self.global_feature = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Linear(512, 1024)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 点云数据 [B, N, 3] 其中N是点的数量
        """
        # 转换为 [B, 3, N] 格式
        x = x.transpose(2, 1)

        # 点特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # 全局最大池化
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # 全局特征
        global_feat = self.global_feature(x)

        return global_feat


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size = x.size(0)

        # 线性变换
        Q = self.w_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 注意力计算
        attention_output = self.scaled_dot_product_attention(Q, K, V)

        # 拼接多头
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        # 输出投影
        output = self.w_o(attention_output)

        return output

    def scaled_dot_product_attention(self, Q, K, V):
        """缩放点积注意力"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)
        return output


class SegmentationHead(nn.Module):
    """分割头"""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # 上采样网络
        self.upconv1 = nn.ConvTranspose2d(feature_dim, 256, 4, 2, 1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.upconv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)

        # 最终分割层
        self.final_conv = nn.Conv2d(32, num_classes, 1)

        # 批归一化
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)

    def forward(
        self, features: torch.Tensor, target_size: tuple[int, int]
    ) -> torch.Tensor:
        """前向传播"""
        # 将特征重塑为2D特征图
        batch_size = features.size(0)
        feature_map = features.view(batch_size, self.feature_dim, 1, 1)

        # 上采样
        x = F.relu(self.bn1(self.upconv1(feature_map)))
        x = F.relu(self.bn2(self.upconv2(x)))
        x = F.relu(self.bn3(self.upconv3(x)))
        x = F.relu(self.bn4(self.upconv4(x)))

        # 最终分割
        x = self.final_conv(x)

        # 调整到目标尺寸
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return x


class DefectDetectionModel:
    """缺陷检测模型包装器"""

    def __init__(self, model_path: str, device: str = 'cuda'):
        """初始化模型"""
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model.eval()

        # 预处理变换
        self.image_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # 缺陷类别映射
        self.defect_classes = [
            'background',
            'scratch',
            'dent',
            'corrosion',
            'inclusion',
            'hole',
        ]

        # 等级类别映射
        self.grade_classes = ['qualified', 'minor', 'major', 'reject']

    def _load_model(self, model_path: str) -> nn.Module:
        """加载模型"""
        try:
            # 加载模型权重
            checkpoint = torch.load(model_path, map_location=self.device)

            # 创建模型实例
            model = MultiModalFusionNetwork()

            # 加载权重
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.to(self.device)
            return model

        except Exception as e:
            raise RuntimeError(f'模型加载失败: {e}')

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB图像
            image_tensor = self.image_transform(image)
        else:
            # 灰度图像
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image_tensor = self.image_transform(image)

        return image_tensor.unsqueeze(0).to(self.device)

    def preprocess_pointcloud(self, pointcloud: np.ndarray) -> torch.Tensor:
        """预处理点云"""
        # 确保点云格式为 [N, 3]
        if pointcloud.shape[1] != 3:
            raise ValueError('点云数据必须是 [N, 3] 格式')

        # 归一化点云
        pointcloud = pointcloud - pointcloud.mean(axis=0)
        pointcloud = pointcloud / (pointcloud.std(axis=0) + 1e-8)

        # 转换为张量
        pointcloud_tensor = torch.from_numpy(pointcloud).float()
        pointcloud_tensor = pointcloud_tensor.unsqueeze(0).to(self.device)

        return pointcloud_tensor

    def predict(self, image: np.ndarray, pointcloud: np.ndarray) -> dict[str, any]:
        """执行预测"""
        with torch.no_grad():
            # 预处理输入
            image_tensor = self.preprocess_image(image)
            pointcloud_tensor = self.preprocess_pointcloud(pointcloud)

            # 模型推理
            outputs = self.model(image_tensor, pointcloud_tensor)

            # 后处理结果
            results = self._postprocess_outputs(outputs, image.shape[:2])

            return results

    def _postprocess_outputs(
        self, outputs: dict[str, torch.Tensor], original_size: tuple[int, int]
    ) -> dict[str, any]:
        """后处理模型输出"""
        # 缺陷检测结果
        detection_probs = F.softmax(outputs['detection'], dim=1)
        torch.argmax(detection_probs, dim=1)

        # 等级评估结果
        grading_probs = F.softmax(outputs['grading'], dim=1)
        grading_pred = torch.argmax(grading_probs, dim=1)

        # 分割结果
        segmentation_probs = F.softmax(outputs['segmentation'], dim=1)
        segmentation_pred = torch.argmax(segmentation_probs, dim=1)

        # 转换为numpy数组
        detection_probs_np = detection_probs.cpu().numpy()[0]
        grading_probs_np = grading_probs.cpu().numpy()[0]
        segmentation_mask = segmentation_pred.cpu().numpy()[0]

        # 调整分割掩码尺寸
        segmentation_mask = cv2.resize(
            segmentation_mask.astype(np.uint8),
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # 提取缺陷信息
        defects = self._extract_defects(segmentation_mask, detection_probs_np)

        return {
            'defects': defects,
            'overall_grade': self.grade_classes[grading_pred.item()],
            'grade_confidence': float(grading_probs_np.max()),
            'segmentation_mask': segmentation_mask,
            'detection_probabilities': detection_probs_np,
            'grading_probabilities': grading_probs_np,
        }

    def _extract_defects(
        self, segmentation_mask: np.ndarray, detection_probs: np.ndarray
    ) -> list[dict[str, any]]:
        """从分割掩码中提取缺陷信息"""
        defects = []

        # 遍历每个缺陷类别
        for class_id in range(1, len(self.defect_classes)):  # 跳过背景类
            class_mask = segmentation_mask == class_id

            if not class_mask.any():
                continue

            # 连通域分析
            num_labels, labels = cv2.connectedComponents(class_mask.astype(np.uint8))

            for label_id in range(1, num_labels):
                defect_mask = labels == label_id

                # 计算缺陷属性
                defect_info = self._analyze_defect(
                    defect_mask, class_id, detection_probs
                )

                if defect_info:
                    defects.append(defect_info)

        return defects

    def _analyze_defect(
        self, defect_mask: np.ndarray, class_id: int, detection_probs: np.ndarray
    ) -> Optional[dict[str, any]]:
        """分析单个缺陷"""
        # 计算缺陷面积
        area = np.sum(defect_mask)

        if area < 10:  # 过滤太小的区域
            return None

        # 获取缺陷边界框
        coords = np.where(defect_mask)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        # 计算缺陷中心
        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2

        # 计算缺陷尺寸
        width = x_max - x_min + 1
        height = y_max - y_min + 1

        return {
            'type': self.defect_classes[class_id],
            'confidence': float(detection_probs[class_id]),
            'area': int(area),
            'center': [int(center_x), int(center_y)],
            'bbox': [int(x_min), int(y_min), int(width), int(height)],
            'dimensions': {'width': int(width), 'height': int(height)},
        }
