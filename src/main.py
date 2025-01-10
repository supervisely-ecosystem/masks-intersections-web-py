import time
import numpy as np
import cv2

from supervisely.app.widgets import Select, Button, Container
from sly_sdk.webpy.app import WebPyApplication
from sly_sdk.sly_logger import logger



select = Select(items=[Select.Item("download", "Download"), Select.Item("extract", "Extract green")], widget_id="select_widget")
button = Button("Extract Mask", widget_id="button_widget")
layout = Container(widgets=[select, button], widget_id="layout")

local_cache = {}
last_geometry_version = {}


# Main script must have object "app" of WebApplication class
app = WebPyApplication(layout=layout)


def download_green(image_id):
    from sly_sdk.api.api import Api

    server_address = app.get_server_address()
    api_token = app.get_api_token()
    api = Api(server_address, api_token, ignore_task_id=True)
    team_id = app.get_team_id()
    api.file.download(team_id, f"/green_mask/{image_id}.png", "/tmp/img.png")
    mask = cv2.imread("/tmp/img.png")
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask.astype(bool)
    return mask


def extract_green(image, smooth=False):
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    if smooth:
        kernel = np.ones((3, 3), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    
    green_mask = green_mask.astype(bool)

    return green_mask


def extract_green_from_figure(green, figure):
    t = time.perf_counter()
    figure_geometry = figure.geometry
    logger.debug("get geometry from figure time: %.4f ms", (time.perf_counter() - t) * 1000)
    x, y = figure_geometry["origin"]
    mask = figure_geometry["data"]
    green = green[y : y + mask.shape[0], x : x + mask.shape[1]]
    mask = mask * green
    return mask


@app.event(app.Event.figure_geometry_saved)
def geometry_updated(event_payload):
    figure_id = event_payload["figureId"]
    t = time.perf_counter()
    figure = app.get_figure_by_id(figure_id)
    logger.debug("get figure time: %.4f ms", (time.perf_counter() - t) * 1000)
    if figure is None:
        return
    current_geom_version = figure.geometry_version
    last_geom_version = last_geometry_version.get(figure_id, None)
    last_geometry_version[figure_id] = current_geom_version + 2
    if last_geom_version is not None and last_geom_version >= current_geom_version:
        return
    green = get_mask()
    t = time.perf_counter()
    mask = extract_green_from_figure(green, figure)
    logger.debug("extract green from figure time: %.4f ms", (time.perf_counter() - t) * 1000)
    t = time.perf_counter()
    # figure = app.update_figures([figure])[0]
    app.update_figure_geometry(figure, mask)
    logger.debug("update figure time: %.4f ms", (time.perf_counter() - t) * 1000)


def get_mask(force = False):
    t = time.perf_counter()
    img_id = app.get_current_image_id()
    logger.debug("get image id time: %.4f ms", (time.perf_counter() - t) * 1000)
    if force or img_id not in local_cache:
        t = time.perf_counter()
        if select.get_value() == "download":
            green = download_green(img_id)
        else:
            img = app.get_current_image()
            green = extract_green(img)
        logger.debug("download mask time: %.4f ms", (time.perf_counter() - t) * 1000)
        local_cache[img_id] = green
    return local_cache[img_id]



@button.click
def save_mask():
    get_mask(force=True)