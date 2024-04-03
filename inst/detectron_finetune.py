# detectron_finetune.py
from detectron2.config import get_cfg
from detectron2.model_zoo import get_config
from detectron2.config import LazyConfig
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets import load_coco_json
from detectron2.data import MetadataCatalog, DatasetCatalog
import os
import yaml


def load_and_filter_dataset(annotations_path, img_path):
    # Load the dataset using Detectron2's COCO JSON loader
    dataset_dicts = load_coco_json(annotations_path, img_path)
    
    # Filter out entries that don't have annotations
    filtered_dataset_dicts = [d for d in dataset_dicts if len(d["annotations"]) > 0]
    
    return filtered_dataset_dicts


def finetune_model(annotations_path, img_path, dataset_name, out_dir, dev, max_iter, num_classes):
  #get_config("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.py")
    # Register the custom dataset
    if dataset_name in DatasetCatalog.list():
      DatasetCatalog.remove(dataset_name)
    
    DatasetCatalog.register(dataset_name, lambda: load_and_filter_dataset(annotations_path, img_path))
    
    cfg = get_cfg()
    cfg.OUTPUT_DIR = out_dir
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.MODEL.DEVICE = dev
    
    # Save the configuration to a config.yaml file
    config_yaml_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with open(config_yaml_path, 'w') as file:
      yaml.dump(cfg, file)
    
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print("Fine-tuning completed.")

