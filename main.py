from functools import partial
from pathlib import Path
from typing import Any, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow._api.v2 import image
from tensorflow.python.data.ops.dataset_ops import TensorSliceDataset
from tensorflow.python.framework.ops import Tensor

from images import BBox, label_img


def add_image_type_to_labels(
    df: pd.DataFrame, p: Path, o: Union[Path, None] = None
) -> pd.DataFrame:
    """
    Read file names form `images` folder and based on which subdir they are
    in, add a label to the label dataframe and save to `o`-path.
    """
    d = {"filename": [], "image_type": []}
    for child_dir in p.iterdir():
        name = child_dir.stem
        l = [i.name for i in child_dir.iterdir()]
        d["image_type"].extend([name for _ in range(len(l))])
        d["filename"].extend(l)
    image_type_df = pd.DataFrame(d)
    new_df = pd.merge(df, image_type_df, how="inner", on="filename")
    if o:
        new_df.to_csv(o, index=False)
    return new_df


def add_dx_dy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a dx and dy column to the dataframe
    """
    df.loc[:, "dx"] = df["xmax"] - df["xmin"]
    df.loc[:, "dy"] = df["ymax"] - df["ymin"]
    return df


def load_dataframe(
    labels_path: Path, image_path: Path, image_type: Union[str, None] = None
) -> pd.DataFrame:
    df = pd.read_csv(labels_path)
    df = add_image_type_to_labels(df, image_path)
    df = add_dx_dy(df)
    if image_type:
        return df.loc[df["image_type"] == image_type]
    return df


def ds_from_df(df: pd.DataFrame) -> TensorSliceDataset:
    return tf.data.Dataset.from_tensor_slices((df["filename"].unique()))


def load_img(
    name_tensor: Tensor,
    box_tensor: Tensor,
    img_name: str,
    size: Tuple[int, int] = (2448, 2448),
) -> Any:
    # TODO: Scale bboxes based on resize or make bboxes between 0 and 1
    labels = tf.boolean_mask(box_tensor, name_tensor == img_name)
    image_path = "data/images/shipcam/" + img_name
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img)
    # img = tf.image.resize(img, size)
    return img, labels


def plot_from_ds(ds: TensorSliceDataset, mapper) -> None:
    plt.figure(figsize=(10, 10))
    for i, (img, labels) in enumerate(
        ds.map(mapper, num_parallel_calls=tf.data.experimental.AUTOTUNE).take(9)
    ):
        bbox = [BBox(*label) for label in labels.numpy()]
        img = label_img(img, bbox)
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(img)  # type: ignore
        ax.set_xticks([])  # type: ignore
        ax.set_yticks([])  # type: ignore
    plt.savefig("tst.png")


if __name__ == "__main__":
    # File paths
    image_path = Path("data", "images")
    labels_path = Path("data", "labels.csv")
    # Create data from with extra info from `labels.csv`
    df = load_dataframe(
        labels_path=labels_path, image_path=image_path, image_type="shipcam"
    )
    # We are building a tf pipeline using tf.Datasets. For this to work we need
    # Some of the info in a mapping function to be tensors. Here we create a tensor
    # With all filenames, and a tensor with all bounding boxes. We will create a
    # boolan mask based on the filename and use that to select the bounding boxes
    # Corresponding to the file
    name_tensor = tf.constant(df["filename"], dtype=tf.string)
    box_tensor = tf.constant(df[["xmin", "ymin", "xmax", "ymax"]], dtype=tf.int32)
    # Here we create a mapping function where we already load in some of the variables
    # so when we are mappen the tf.Dataset we do not have to parse these.
    image_label_loader = partial(load_img, name_tensor, box_tensor)
    # Create a simple tf.Dataset from a dataframe
    ds = ds_from_df(df)
    # Plot the images with labels
    plot_from_ds(ds, image_label_loader)
