# detectron_predict.py
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
import numpy as np
import os, json, cv2, random
import torch, detectron2

import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json
from matplotlib import pyplot as plt
import yaml


def predict_image(image_path, file_path='Detectron2_Models/config.yaml'):
    image = cv2.imread(image_path)
    cfg = get_cfg()
    cfg.merge_from_file(file_path)
    cfg.MODEL.DEVICE = "cpu" #mps
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)
    return outputs

def visualize_output(image_path, outputs):
    image = cv2.imread(image_path)
    v = Visualizer(image[:, :, ::-1], scale=0.5, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()
