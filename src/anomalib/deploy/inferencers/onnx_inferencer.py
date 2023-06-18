"""This module contains ONNX inference implementations."""

# Copyright (C) 2023 Nobuo Tsukamoto
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from omegaconf import DictConfig
import onnxruntime as rt

from anomalib.data import TaskType

from .base_inferencer import Inferencer


class ONNXInferencer(Inferencer):
    """ONNX implementation for the inference.

    Args:
        path (str): Path to the onnx file.
        metadata (str | Path | dict, optional): Path to metadata file or a dict object defining the
            metadata. Defaults to None.
        input_shape (tuple): Model input shape (height, width).
        task (TaskType | None, optional): Task type. Defaults to None.
    """

    def __init__(
        self,
        path: str,
        metadata: str | Path | dict | None = None,
        input_shape: tuple | None = [256, 256],
        task: str | None = None,
    ) -> None:
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.input_shape = input_shape
        self.model = self.load_model(path)
        self.first_input_name = self.model.get_inputs()[0].name
        self.first_output_name = self.model.get_outputs()[0].name

        self.metadata = super()._load_metadata(metadata)

        self.task = TaskType(task) if task else TaskType(self.metadata["task"])

    def load_model(self, path: str | Path):
        """Load the Onnx model.

        Args:
            path (str | Path | tuple[bytes, bytes]): Path to the onnx files

        Returns:
            ONNX model.
        """
        session = rt.InferenceSession(path)
        return session

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        """Pre process the input image by applying transformations.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: pre-processed image.
        """
        processed_image = cv2.resize(image, self.input_shape)
        processed_image = ((processed_image / 255 - self.mean) / self.std).astype(np.float32)

        if len(processed_image.shape) == 3:
            processed_image = np.expand_dims(processed_image, axis=0)

        if processed_image.shape[-1] == 3:
            processed_image = processed_image.transpose(0, 3, 1, 2)

        return processed_image

    def forward(self, image: np.ndarray) -> np.ndarray:
        """Forward-Pass input tensor to the model.

        Args:
            image (np.ndarray): Input tensor.

        Returns:
            np.ndarray: Output predictions.
        """
        return self.model.run([self.first_output_name], {self.first_input_name: image})

    def post_process(self, predictions: np.ndarray, metadata: dict | DictConfig | None = None) -> dict[str, Any]:
        """Post process the output predictions.

        Args:
            predictions (np.ndarray): Raw output predicted by the model.
            metadata (Dict, optional): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.
                Defaults to None.

        Returns:
            dict[str, Any]: Post processed prediction results.
        """
        if metadata is None:
            metadata = self.metadata

        predictions = np.array(predictions)

        # Initialize the result variables.
        anomaly_map: np.ndarray | None = None
        pred_label: float | None = None
        pred_mask: float | None = None

        # If predictions returns a single value, this means that the task is
        # classification, and the value is the classification prediction score.
        if len(predictions.shape) == 1:
            task = TaskType.CLASSIFICATION
            pred_score = predictions
        else:
            task = TaskType.SEGMENTATION
            anomaly_map = predictions.squeeze()
            pred_score = anomaly_map.reshape(-1).max()

        # Common practice in anomaly detection is to assign anomalous
        # label to the prediction if the prediction score is greater
        # than the image threshold.
        if "image_threshold" in metadata:
            pred_label = pred_score >= metadata["image_threshold"]

        if task == TaskType.CLASSIFICATION:
            _, pred_score = self._normalize(pred_scores=pred_score, metadata=metadata)
        elif task in (TaskType.SEGMENTATION, TaskType.DETECTION):
            if "pixel_threshold" in metadata:
                pred_mask = (anomaly_map >= metadata["pixel_threshold"]).astype(np.uint8)

            anomaly_map, pred_score = self._normalize(
                pred_scores=pred_score, anomaly_maps=anomaly_map, metadata=metadata
            )
            assert anomaly_map is not None

            if "image_shape" in metadata and anomaly_map.shape != metadata["image_shape"]:
                image_height = metadata["image_shape"][0]
                image_width = metadata["image_shape"][1]
                anomaly_map = cv2.resize(anomaly_map, (image_width, image_height))

                if pred_mask is not None:
                    pred_mask = cv2.resize(pred_mask, (image_width, image_height))
        else:
            raise ValueError(f"Unknown task type: {task}")

        if self.task == TaskType.DETECTION:
            pred_boxes = self._get_boxes(pred_mask)
            box_labels = np.ones(pred_boxes.shape[0])
        else:
            pred_boxes = None
            box_labels = None

        return {
            "anomaly_map": anomaly_map,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "pred_mask": pred_mask,
            "pred_boxes": pred_boxes,
            "box_labels": box_labels,
        }

    @staticmethod
    def _get_boxes(mask: np.ndarray) -> np.ndarray:
        """Get bounding boxes from masks.

        Args:
            masks (np.ndarray): Input mask of shape (H, W)

        Returns:
            np.ndarray: array of shape (N, 4) containing the bounding box coordinates of the objects in the masks
            in xyxy format.
        """
        _, comps = cv2.connectedComponents(mask)

        labels = np.unique(comps)
        boxes = []
        for label in labels[labels != 0]:
            y_loc, x_loc = np.where(comps == label)
            boxes.append([np.min(x_loc), np.min(y_loc), np.max(x_loc), np.max(y_loc)])
        boxes = np.stack(boxes) if boxes else np.empty((0, 4))
        return boxes
