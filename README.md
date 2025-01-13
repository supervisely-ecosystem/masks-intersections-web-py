<div align="center" markdown>

# Masks intersection

<img src="https://github.com/user-attachments/assets/d5e708a4-d09d-456d-b6f7-065718bf2d34"/>

</div>

This application allows you to modify each bitmap mask created in labeling tool by intersecting it with predefined mask.
The mask should be uploaded to teamfiles by the following path `/green_mask/{image_id}.png` where `{image_id}` is the id of the image you are working with.

If you want to fork the application and add your own logic, please consider the following:
-   Latest version of sly_sdk can be found in this repository: https://github.com/supervisely-ecosystem/client_side_app_template. You can also find the documentation there.
-   Do not modify the contents of the sly_sdk folder.
-   You can modify the contents of the src folder.
-   You can add new modules and import them in the main.py file.
-   If you want to use packages that are not present in the `sly_sdk/requirements.txt` file, you can add them to the `requirements.txt` file in the root of the repository. Only pure python packages are supported plus packages that listed here: https://pyodide.org/en/0.25.0/usage/packages-in-pyodide.html
-   You can create the layout of the application using widgets from the supervisely SDK. But only limited number of widgets are supported at the moment. You can see the list of supported widgets in the sly_sdk.app.widgets folder.
-   Another requirement for using widgets is to set the widgets id yourself. The id should be unique for each widget in the application. Also it is advised to declare the widgets as it is currently done in the `src/main.py` file.

Example:
```python
    my_widget = Text("My Widget", widget_id="layout")
```


# Web Python Applications
You can create a python application that will be running in the Supervisely Labeling tool in the browser. This allows you to have direct access to the objects of the tool such as images and annotations and to create event handlers for the tool.


## How to develop:
Your application should have a file with a `WebPyApplication` object named `app`.
Every application repository should have a sly_sdk module from this repository. This module is required for the application to work. It is temporary and will be removed in the future as we will update Supervisely SDK.

`WebPyApplication` object should be miported from sly_sdk.webpy
Widgets should be imported from `supervisely.app.widgets` but all of them should be available in the `sly_sdk.app.widgets` module as well.
We will add more widgets in the future.


### GUI:
To create a GUI for the application, you need to pass a `layout` argument to the `WebPyApplication` constructor. The layout should be a `Widget` object.
Not all Supervisely widgets are supported yet.


### config.json
In config.json you need to set the following variables:
- `gui_folder_path`: Path from where the GUI will be served. You can enter any non-conflicting path.
- `src_dir`: Path to the directory where the source code is located. All the modules that are imported in the main file should be in this directory.
- `main_script`: Path to the main script file. This file should be in the `src_dir` directory or in a subdirectory of it. This file should contain a variable `app` that is `WebPyApplication` object.

example:
```json
{
    "gui_folder_path": "app",
    "src_dir": "src",
    "main_script": "src/main.py"
}
```

### Events:

You can subscribe to events in the Labeling tool. To do this, you need to create a function and decorate it with the `@app.event` decorator. The function should expect a single argument, which will be the data sent by the event.

Available events can be found in the `app.Event` class

example:
```python
@app.event(app.Event.FigureGeometrySaved)
def on_figure_geometry_saved(event: WebPyApplication.Event.FigureGeometrySaved):
    print("Figure geometry saved:", event.figure_id)
```
