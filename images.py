from typing import List
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np

from dataclasses import dataclass


@dataclass
class BBox:
    """
    Bounding Box dataclass
    """

    xmin: int
    ymin: int
    xmax: int
    ymax: int


def label_img(img: np.ndarray, bbox: List[BBox]) -> None:
    # TODO: move config somewhere else

    # Config, move somewhere else
    fontsize = 40
    margin = 5
    # Load font from assets folder
    fnt = ImageFont.truetype("assets/SourceCodePro-Regular.ttf", 40)
    img = Image.fromarray(img.numpy().astype(np.uint8))  # type: ignore
    # Draw boundingboxes and class labels on the image
    for bb in bbox:
        # Create Draw object
        draw = ImageDraw.Draw(img)
        # Set text position
        text_position = [bb.xmin, bb.ymin - fontsize - margin]
        # Get bounding box of text
        tbb = BBox(*draw.textbbox(text_position, "tst", font=fnt))  # type: ignore
        # Add some vertical margin to text label box
        text_bbox = (tbb.xmin, tbb.ymin - margin, tbb.xmax, tbb.ymin + margin)
        # Draw text background
        draw.rectangle(text_bbox, fill="green")  # type: ignore
        # Draw text
        draw.text(text_position, "tst", font=fnt, fill="black")  # type: ignore
        # Draw bounding box
        draw.rectangle([bb.xmin, bb.ymin, bb.xmax, bb.ymax], outline="green", width=4)  # type: ignore
        del draw
    return img
