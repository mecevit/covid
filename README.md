# Ango Mask Detection Open Dataset

[![Open in Layer](https://app.layer.ai/assets/badge.svg)](https://app.layer.ai/layer/covid) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tH5YvnSTOnAi7kAmv9LudrpX5glb5zZG?usp=sharing)

Our workforce has labeled over four hundred images of people, detecting their faces, what direction the face was facing, whether the faces contained a mask, and whether or not the mask was used correctly. We also trained a model for demo purposes.

Both the images, the labels and the model are free for you to use in your own projects.

## How to train

First install the requirements:
```
pip install -r requirements.txt
```

Get your Layer API from [app.layer.ai](https://app.layer.ai) > Settings > Developer. Set your environment variable:
```
export LAYER_API_KEY=[YOUR_API_KEY]
```

Train your model:
```
python main.py
```

## Dataset

Each image has been independently labeled by two of our labelers, and then manually reviewed by one of our expert reviewers. Our labelers combined annotated a total of 7212 faces, 4819 of which were looking straight, with the rest evenly split between looking to the side and away.

Among the 7212 mask labels, 5079 were wearing their mask correctly, 200 incorrectly, and in the
rest of the cases the mask was either off or not visible.

Sample Annotation:

https://app.layer.ai/layer/covid/datasets/mask_images?tab=logs#sample-1

Here is the dataset:

https://app.layer.ai/layer/covid/datasets/mask_images

## Model

We used the dataset to train a object detection model. Model is a modified [Faster RCNN](https://pytorch.org/vision/stable/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html). With just 5 epochs, we can get very satisfying results:

https://app.layer.ai/layer/covid/models/mask_predictor#prediction

Here is the model:

https://app.layer.ai/layer/covid/models/mask_predictor