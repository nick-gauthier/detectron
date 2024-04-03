detectron_train <- function(annotations_path, img_path, dataset_name="my_dataset_train", out_dir="Detectron2_Models", dev='cpu', max_iter=1000, num_classes=3L) {

  # Load the detectron2 module
  reticulate::source_python('inst/detectron_finetune.py')

  # Fine-tune the model
  finetune_model(annotations_path, img_path, dataset_name, out_dir, dev, max_iter, num_classes)

}

