import time
import numpy as np
import cv2

from supervisely.app.widgets import Container, Button
from sly_sdk.webpy import WebPyApplication


button = globals().get("button", Button("Extract green", widget_id="widget_3"))
layout = globals().get("layout", Container(widgets=[button], widget_id="widget_4"))

local_cache = {}
figures_versions = {}


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

    return green_mask

def extract_green_from_figure(green, figure):
    x, y = figure.geometry["origin"]
    mask = figure.geometry["data"]
    green = green[y:y+mask.shape[0], x:x+mask.shape[1]]
    mask = np.where((green == 255) & (mask == 255), 255, 0)
    return figure.clone(geometry={"origin": (x, y), "data": mask})


@app.event(app.Event.figure_geometry_changed)
def geometry_updated(event_payload):
    print("geometry_updated")
    print("event_payload", event_payload)
    figure_id = event_payload["figureId"]
    t = time.perf_counter()
    figure = app.get_figure_by_id(figure_id)
    print("get figure time", time.perf_counter() - t)
    if figure is None:
        return
    last_version = figures_versions.get(figure_id, None)
    if last_version is not None and last_version >= figure.version:
        return
    figures_versions[figure_id] = figure.version + 1
    t = time.perf_counter()
    img_id = app.get_current_image_id()
    print("get image id time", time.perf_counter() - t)
    if img_id not in local_cache:
        t = time.perf_counter()
        green = download_green(img_id)
        print("download mask time", time.perf_counter() - t)
        local_cache[img_id] = green
    else:
        green = local_cache[img_id]
    t = time.perf_counter()
    figure = extract_green_from_figure(green, figure)
    print("extract green from figure time", time.perf_counter() - t)
    t = time.perf_counter()
    figure = app.update_figures([figure])[0]
    print("update figure time", time.perf_counter() - t)
    print("updated figure. version:", figure.version)


@button.click
def on_button_click():
    print("Button clicked!")
    img_id = app.get_current_image_id()
    if img_id not in local_cache:
        green = download_green(img_id)
        local_cache[img_id] = green
    else:
        green = local_cache[img_id]
    figure = app.get_selected_figure()
    if figure is None:
        return
    figure = extract_green_from_figure(green, figure)
    return app.update_figures([figure])

@app.run_function
def run_default(*args, **kwargs):
    print("args", [f"{i}. {arg}" for i, arg in enumerate(args, 1)])
    print("kwargs", kwargs)