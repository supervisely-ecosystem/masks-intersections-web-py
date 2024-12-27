import asyncio
import enum
import io
import json
import tarfile
from typing import List, NamedTuple, Optional
import cv2
from fastapi import FastAPI
import numpy as np


def get_or_create_event_loop():
    """
    Get the current event loop or create a new one if it doesn't exist.
    Works for different Python versions and contexts.

    :return: Event loop
    :rtype: asyncio.AbstractEventLoop
    """
    import asyncio

    try:
        # Preferred method for asynchronous context (Python 3.7+)
        return asyncio.get_running_loop()
    except RuntimeError:
        # If the loop is not running, get the current one or create a new one (Python 3.8 and 3.9)
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            # For Python 3.10+ or if the call occurs outside of an active loop context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop


def await_async(coro):
    loop = asyncio.get_event_loop()
    res = loop.run_until_complete(coro)
    return res

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        local = kwargs.pop("__local__", False)
        if local is False:
            if cls not in cls._instances:
                cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
            return cls._instances[cls]
        else:
            return super(Singleton, cls).__call__(*args, **kwargs)


class Field(str, enum.Enum):
    STATE = "state"
    DATA = "data"
    CONTEXT = "context"


class FigureInfo(NamedTuple):
    id: int
    class_id: int
    updated_at: str
    created_at: str
    entity_id: int
    object_id: int
    project_id: int
    dataset_id: int
    frame_index: int
    geometry_type: str
    geometry: dict
    geometry_meta: dict
    tags: list
    meta: dict
    area: str
    priority: Optional[int] = None
    version: int = 0

    def clone(self, **kwargs):
        return self._replace(**kwargs)


def base64_2_data(s: str) -> np.ndarray:
    import base64
    from PIL import Image
    import zlib
    try:
        z = zlib.decompress(base64.b64decode(s))
    except zlib.error:
        # If the string is not compressed, we'll not use zlib.
        img = Image.open(io.BytesIO(base64.b64decode(s)))
        return np.array(img)
    n = np.frombuffer(z, np.uint8)

    imdecoded = cv2.imdecode(n, cv2.IMREAD_GRAYSCALE)  # pylint: disable=no-member
    if (len(imdecoded.shape) == 3) and (imdecoded.shape[2] == 4):
        mask = imdecoded[:, :, 3]  # pylint: disable=unsubscriptable-object
    if (len(imdecoded.shape) == 3) and (imdecoded.shape[2] == 1):
        mask = imdecoded[:, :, 0] # pylint: disable=unsubscriptable-object
    elif len(imdecoded.shape) == 2:
        mask = imdecoded
    else:
        raise RuntimeError("Wrong internal mask format.")
    return mask


def get_figure_data(js_figure):
    img_cvs = js_figure._geometry._main.bitmap
    img_ctx = img_cvs.getContext("2d")
    img_data = img_ctx.getImageData(0, 0, img_cvs.width, img_cvs.height).data
    img_data = np.array(img_data, dtype=np.uint8).reshape(img_cvs.height, img_cvs.width, 4)
    rgb = img_data[:, :, :3]
    alpha = img_data[:, :, 3]
    return rgb, alpha


def put_img_to_figure(js_figure, img_data: np.ndarray):
    from js import ImageData, console
    from pyodide.ffi import create_proxy

    img_data = img_data.flatten().astype(np.uint8)
    pixels_proxy = create_proxy(img_data)
    pixels_buf = pixels_proxy.getBuffer("u8clamped")
    img_cvs = js_figure._geometry._main.bitmap
    new_img_data = ImageData.new(pixels_buf.data, img_cvs.width, img_cvs.height)
    img_ctx = img_cvs.getContext("2d")
    img_ctx.putImageData(new_img_data, 0, 0)
    pixels_proxy.destroy()
    pixels_buf.release()


def py_to_js(obj):
    from pyodide.ffi import to_js
    from js import Object

    if isinstance(obj, dict):
        js_obj = Object()
        for key, value in obj.items():
            setattr(js_obj, key, py_to_js(value))
        return js_obj
    elif isinstance(obj, list):
        return [py_to_js(item) for item in obj]
    else:
        return to_js(obj)

def js_to_py(obj):
    return obj.to_py()


class _PatchableJson(dict):
    def __init__(self, field: Field, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._field = field
        self._linked_obj = None

    def raise_for_key(self, key: str):
        if key in self:
            raise KeyError(f"Key {key} already exists in {self._field}")

    def __update(self, js_obj):
        self.update(js_obj.to_py())

    def link(self, js_obj):
        self._linked_obj = js_obj
        self.__update(js_obj)

    def send_changes(self):
        if self._linked_obj is None:
            return

        for key, value in self.items():
            setattr(self._linked_obj, key, py_to_js(value))


class StateJson(_PatchableJson, metaclass=Singleton):
    def __init__(self, *args, **kwargs):
        super().__init__(Field.STATE, *args, **kwargs)


class DataJson(_PatchableJson, metaclass=Singleton):
    def __init__(self, *args, **kwargs):
        super().__init__(Field.DATA, *args, **kwargs)


class MainServer(metaclass=Singleton):
    def __init__(self):
        self._server = FastAPI()

    def get_server(self) -> FastAPI:
        return self._server


# SDK code
class WebPyApplication(metaclass=Singleton):
    def __init__(self):
        self._run_f = None
        self._widgets_n = 0
        self.is_inited = False

    def __init_state(self):
        from js import slyApp

        self._slyApp = slyApp
        app = slyApp.app
        app = getattr(app, "$children")[0]  # <- based on template
        self._state = app.state
        self._data = app.data
        self._context = app.context  # ??
        self._store = slyApp.store  # <- Labeling tool store (image, classes, objects, etc)

        StateJson().link(self._state)
        DataJson().link(self._data)

        self.is_inited = True

    # Labeling tool data access
    def get_current_image(self):
        cur_img = getattr(self._store.state.videos.all, str(self._context.imageId))
        img_src = cur_img.sources[0]
        img_cvs = img_src.imageData
        img_ctx = img_cvs.getContext("2d")
        img_data = img_ctx.getImageData(0, 0, img_cvs.width, img_cvs.height).data
        img_arr = np.array(img_data, dtype=np.uint8).reshape((img_cvs.height, img_cvs.width, 4))
        return img_arr

    def get_current_image_id(self):
        return self._context.imageId

    def get_selected_figure(self):
        js_figure = self._store.getters.as_object_map()

    def get_current_view_figures(self):
        from pyodide.webloop import PyodideFuture
        import js

        js_figures = self._store.getters.as_object_map()["figures/currentViewFigures"]
        if isinstance(js_figures, PyodideFuture):
            js_figures = await_async(js_figures)
        
        figures: List[FigureInfo] = []
        for js_figure in js_figures:
            if js_figure._geometryType != "bitmap":
                print(f"Only bitmaps supported at the moment, skipping object #{js_figure.id}")
                continue
            rgb, alpha = get_figure_data(js_figure)
            offset = (js_figure._geometry._main.offset.x, js_figure._geometry._main.offset.y)
            figures.append(FigureInfo(
                id=js_figure.id,
                class_id=js_figure.classId,
                updated_at=js_figure.updatedAt,
                created_at= js_figure.createdAt,
                entity_id=None,
                object_id=js_figure.objectId,
                project_id=None,
                dataset_id=None,
                frame_index=None,
                geometry_type=js_figure._geometryType,
                geometry={"data": alpha, "origin": offset},
                geometry_meta=None,
                tags=js_to_py(js_figure.tags),
                meta=js_to_py(js_figure.meta),
                area=js_figure.area,
                priority=js_figure.priority,
                version=js_figure.version
            ))
        return figures

    def update_figures(self, figures: List[FigureInfo]):
        from pyodide.webloop import PyodideFuture
        from pyodide.ffi import to_js
        import js

        js_figures = self._store.getters.as_object_map()["figures/currentViewFigures"]
        if isinstance(js_figures, PyodideFuture):
            js_figures = await_async(js_figures)

        for js_figure in js_figures:
            for figure in figures:
                if figure.id == js_figure.id:
                    self._store.dispatch('figures/figureGeometryBeforeUpdate', figure.id)
                    # rgb, alpha = get_figure_data(js_figure)
                    img = np.stack([figure.geometry["data"]] * 4, axis=-1)
                    put_img_to_figure(js_figure, img)
                    self._store.dispatch('figures/updateGeometryInFigure', to_js({
                        "figureId": figure.id,
                        "commit": True,
                        "data": {
                            "version": figure.version + 1,
                        },
                    }, dict_converter=js.Object.fromEntries))

    @property
    def state(self):
        if not self.is_inited:
            self.__init_state()
        StateJson().link(self._state)
        return StateJson()

    @property
    def data(self):
        if not self.is_inited:
            self.__init_state()
        DataJson().link(self._data)
        return DataJson()

    @classmethod
    def render(cls, layout, src_dir="", app_dir="", requirements_path: str = None):
        import json
        import os
        from pathlib import Path
        import shutil
        import supervisely as sly
        from supervisely.app.content import DataJson, StateJson
        from fastapi.staticfiles import StaticFiles
        from fastapi.routing import Mount

        app = sly.Application(layout=layout)
        reqs = None
        if requirements_path is not None:
            reqs = Path(requirements_path).read_text().splitlines()
        index = app.render({"__webpy_script__": True, "pyodide_requirements": reqs})
        # @change="post('/{{{widget.widget_id}}}/value_changed')" -> @change="runPythonScript('/{{{widget.widget_id}}}/value_changed')"
        index = index.replace("post('/", "runPythonScript('/")

        src_dir = Path(src_dir)
        app_dir = Path(app_dir)
        os.makedirs(app_dir, exist_ok=True)
        with open(app_dir / "index.html", "w") as f:
            f.write(index)

        json.dump(StateJson(), open(app_dir / "state.json", "w"))
        json.dump(DataJson(), open(app_dir / "data.json", "w"))

        shutil.copy(src_dir / "gui.py", app_dir / "gui.py")
        shutil.copy(src_dir / "main.py", app_dir / "main.py")

        with tarfile.open(app_dir / "sly_sdk.tar", "w") as tar:
            tar.add(
                "sly_sdk",
                arcname="sly_sdk",
                filter=lambda tarinfo: (
                    None
                    if "__pycache__" in tarinfo.name or tarinfo.name.endswith(".pyc")
                    else tarinfo
                ),
            )

        server = app.get_server()
        for route in server.routes:
            if route.path == "/sly":
                route: Mount
                for route in route.routes:
                    if route.path == "/css" and isinstance(route.app, StaticFiles):
                        source_dir = route.app.directory
                        for root, _, files in os.walk(source_dir):
                            rel_path = Path(root).relative_to(source_dir)
                            for file in files:
                                if file.endswith(("css", "js", "html")):
                                    sly.fs.copy_file(
                                        Path(root, file), app_dir / Path("sly/css", rel_path, file)
                                    )

    def _get_handler(self, *args, **kwargs):
        if len(args) != 1:
            return None
        if not isinstance(args[0], str):
            return None
        handlers = kwargs.get("widgets_handlers", {})
        if args[0] in handlers:
            return handlers[args[0]]
        return None

    def _run_handler(self, f, *args, **kwargs):
        import inspect

        if inspect.iscoroutinefunction(f):
            loop = get_or_create_event_loop()
            return loop.run_until_complete(f(*args, **kwargs))
        return f(*args, **kwargs)

    def run(self, *args, **kwargs):
        import gui
        from fastapi.routing import APIRoute
        from js import console

        self.state
        self.data  # to init StateJson and DataJson

        server = MainServer().get_server()
        handlers = {}
        for route in server.router.routes:
            if isinstance(route, APIRoute):
                handlers[route.path] = route.endpoint

        handler = self._get_handler(*args, widgets_handlers=handlers, **kwargs)
        if handler is not None:
            return self._run_handler(handler)
        if self._run_f is None:
            print("Unknown command")
        return self._run_f(*args, **kwargs)

    def run_function(self, f):
        self._run_f = f
        return f
