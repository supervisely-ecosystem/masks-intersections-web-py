from supervisely.app.widgets import Container, Button


button = globals().get("button", Button("Click me!", widget_id="widget_3"))
layout = globals().get("layout", Container(widgets=[button], widget_id="widget_4"))
