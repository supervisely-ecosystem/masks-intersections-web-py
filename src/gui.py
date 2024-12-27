import numpy as np
from supervisely.app.widgets import Container, Button
from sly_sdk.sly import WebPyApplication


button = globals().get("button", Button("Extract green", widget_id="widget_3"))
layout = globals().get("layout", Container(widgets=[button], widget_id="widget_4"))

app = WebPyApplication()


def extract_green(image, smooth=False):
    import cv2
    import numpy as np

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    if smooth:
        kernel = np.ones((3, 3), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    return green_mask


def extract_green_from_figure(img, figure):
    x, y = figure.geometry["origin"]
    mask = figure.geometry["data"]
    green = extract_green(image=img[x:x+mask.shape[0], y:y+mask.shape[1]])
    mask = np.where((green == 255) & (mask == 255), 255, 0)
    return figure.clone(geometry={"origin": (x, y), "data": mask})

@button.click
def on_button_click():
    print("Button clicked!")
    img = app.get_current_image()
    figures = app.get_current_view_figures()
    updated_figures = []
    for figure in figures:
        updated_figures.append(extract_green_from_figure(img, figure))
    app.update_figures(updated_figures)
