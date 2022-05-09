import layer
from layer.decorators import model, fabric, pip_requirements
import utils
import coco_utils
import coco_eval
import engine
from engine import train_one_epoch, evaluate
import transforms
import cloudpickle
import dataset
from dataset import AngoDataset
from utils import collate_fn
import os

def get_instance_segmentation_model(num_classes):
  import torchvision
  from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  model.roi_heads.mask_predictor = None
  return model


def get_transform(is_train):
  transformations = []
  transformations.append(transforms.ToTensor())
  if is_train:
    transformations.append(transforms.RandomHorizontalFlip(0.5))
  return transforms.Compose(transformations)


def get_datasets():
  from torch.utils.data import DataLoader
  import layer
  import torch

  df = layer.get_dataset('layer/covid/datasets/mask_images:1.1').to_pandas()
  dataset = AngoDataset(df, get_transform(True))
  dataset_test = AngoDataset(df, get_transform(False))

  torch.manual_seed(1)
  indices = torch.randperm(len(dataset)).tolist()
  dataset = torch.utils.data.Subset(dataset, indices[:-50])
  dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

  data_loader = DataLoader(
    dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

  data_loader_test = DataLoader(
    dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

  return data_loader, data_loader_test


@model("mask_predictor")
@fabric("f-gpu-small")
@pip_requirements(packages=["torchvision", "pycocotools"])
def train():
  import torchvision
  import torch
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  data_loader, data_loader_test = get_datasets()

  layer.log({
    "train_size": len(data_loader.dataset),
    "test_size": len(data_loader_test.dataset)
  })

  num_classes = 3
  model = get_instance_segmentation_model(num_classes)
  model.to(device)

  params = [p for p in model.parameters() if p.requires_grad]
  parameters = {
    "lr": 0.005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "epochs": 5,
    "step_size": 3,
    "gamma": 0.1
  }
  layer.log(parameters)

  optimizer = torch.optim.SGD(params, lr=parameters["lr"], momentum=parameters["momentum"],
                              weight_decay=parameters["weight_decay"])
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=parameters["step_size"],
                                                 gamma=parameters["gamma"])

  num_epochs = parameters["epochs"]
  step = 0
  for epoch in range(num_epochs):
    step = train_one_epoch(model, optimizer, data_loader, device, epoch, 1, step)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device=device)

  model.eval()
  model.to(device)
  with torch.no_grad():
    img, _ = data_loader_test.dataset[16]
    prediction = model([img.to(device)])
    img = torch.tensor(img.mul(255), dtype=torch.uint8)
    score_threshold = .9
    scores = prediction[0]['scores']
    boxes = prediction[0]['boxes'][scores > score_threshold]
    labels = prediction[0]['labels']
    labels = labels[scores > score_threshold]
    colors = [None, (0, 255, 0), (255, 0, 0)]
    col = [colors[label] for label in labels]
    labels = scores[scores > score_threshold]
    labels = list(map("{:.3f}".format, labels.tolist()))

    bboxes = torchvision.utils.draw_bounding_boxes(img, boxes, colors=col, labels=labels, width=3, fill=True)
    predict_masks = torchvision.transforms.ToPILImage()(bboxes)
    layer.log({"prediction": predict_masks})

  return model

# Register all modules in the project
cloudpickle.register_pickle_by_value(utils)
cloudpickle.register_pickle_by_value(dataset)
cloudpickle.register_pickle_by_value(coco_eval)
cloudpickle.register_pickle_by_value(coco_utils)
cloudpickle.register_pickle_by_value(transforms)
cloudpickle.register_pickle_by_value(engine)

token = os.getenv("LAYER_API_KEY")
layer.login_with_api_key(token)
layer.init("covid")

layer.run([train])
