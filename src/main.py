try:
    import supervisely
except ImportError:
    import sys

    import sly_sdk as supervisely

    sys.modules["supervisely"] = supervisely


from sly_sdk.sly import WebPyApplication

app = WebPyApplication()


app.run
