from supervisely.app.widgets import Container, Button
from sly_sdk.sly import WebPyApplication


button = globals().get("button", Button("Click me!", widget_id="widget_3"))
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


@button.click
def on_button_click():
    print("Button clicked!")
    img = app.get_current_image()
    green_mask = extract_green(img)
    app.get_figures()
