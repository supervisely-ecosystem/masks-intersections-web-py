from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from sly_sdk.render import render

gui_dir = render()

app = FastAPI()

app.mount("/", StaticFiles(directory=gui_dir))
