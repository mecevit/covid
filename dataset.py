import torch


class AngoDataset(torch.utils.data.Dataset):

  def __init__(self, df, transforms=None):
    self.df = df
    self.transforms = transforms

  def __getitem__(self, idx):
    from io import BytesIO
    from base64 import b64decode
    import numpy as np
    from PIL import Image
    import torch.utils.data

    row = self.df.iloc[idx]
    img = Image.open(BytesIO(b64decode(row.image.encode("utf-8"))))
    # size = (512,512)
    # img.thumbnail(size, Image.ANTIALIAS)
    img = img.convert("RGB")
    dw, dh = img.size
    masks_array = np.array(row.boxes).reshape(int(len(row.boxes) / 5), -1)
    boxes = []
    labels = []
    for boxdef in masks_array:
      width = float(boxdef[3]) * dw
      height = float(boxdef[4]) * dh
      xmin = float(boxdef[1]) * dw - width / 2
      ymin = float(boxdef[2]) * dh - height / 2
      xmax = xmin + width
      ymax = ymin + height
      boxes.append([xmin, ymin, xmax, ymax])
      labels.append(int(boxdef[0])+1)

    num_objs = len(masks_array)
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    image_id = torch.tensor([int(row.id)])
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    is_crowd = torch.zeros((num_objs,), dtype=torch.int64)

    target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": is_crowd}

    if self.transforms is not None:
      img, target = self.transforms(img, target)

    return img, target

  def __len__(self):
    return self.df.shape[0]
