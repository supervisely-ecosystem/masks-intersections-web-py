from __future__ import annotations
from typing import List


from sly_sdk.webpy import DataJson, StateJson, MainServer, Singleton


# def rand_str(length):
#     chars = string.ascii_letters + string.digits  # [A-z][0-9]
#     return "".join((random.choice(chars)) for _ in range(length))


#
#  DELETED BECAUSE SHOULD NOT BE USED IN WEBPY RUNTIME
#
# def create_env(directory: Union[str, os.PathLike]) -> "jinja2.Environment":
#     env_sly = jinja2.Environment(
#         loader=jinja2.FileSystemLoader(directory),
#         autoescape=True,
#         variable_start_string="{{{",
#         variable_end_string="}}}",
#     )
#     return env_sly


# class JinjaWidgets(metaclass=Singleton):
#     def __init__(self, auto_widget_id=True):
#         self.auto_widget_id = auto_widget_id
#         self.context = {}
#         self.context["__no_html_mode__"] = auto_widget_id
#         self.context["__widget_scripts__"] = {}


class BaseWidget:
    def __init__(self, widget_id: str):
        super().__init__()
        self.server = MainServer()
        self.widget_id = widget_id


class Hidable:
    def __init__(self):
        self._hide = False
        if not hasattr(self, "widget_id"):
            self.widget_id = None

    def is_hidden(self):
        return self._hide

    def hide(self):
        self._hide = True
        DataJson()[self.widget_id]["hide"] = self._hide
        DataJson().send_changes()

    def show(self):
        self._hide = False
        DataJson()[self.widget_id]["hide"] = self._hide
        DataJson().send_changes()

    def get_json_data(self):
        return {"hide": self._hide}

    def get_json_state(self):
        return {}

    #
    #  DELETED BECAUSE SHOULD NOT BE USED IN WEBPY RUNTIME
    #
    # def _wrap_hide_html(self, html):
    #     soup = BeautifulSoup(html, features="html.parser")
    #     for item in soup:
    #         if hasattr(item.__class__, "has_attr") and callable(
    #             getattr(item.__class__, "has_attr")
    #         ):
    #             if item.has_attr("v-if"):
    #                 item["v-if"] = f'({item["v-if"]}) && data.{self.widget_id}.hide === false'
    #             else:
    #                 item["v-if"] = f"!data.{self.widget_id}.hide"
    #     # return f'<div v-if="!data.{widget_id}.hide">{html}</div>'
    #     return str(soup)

    # def wrap(self, html):
    #     return self._wrap_hide_html(html)


class Disableable:
    def __init__(self):
        self._disabled = False
        if not hasattr(self, "widget_id"):
            self.widget_id = None

    def is_disabled(self):
        return self._disabled

    def disable(self):
        self._disabled = True
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def enable(self):
        self._disabled = False
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def get_json_data(self):
        return {"disabled": self._disabled}

    def get_json_state(self):
        return {}

    #
    #  DELETED BECAUSE SHOULD NOT BE USED IN WEBPY RUNTIME
    #
    # def _wrap_disable_html(self, html):
    #     soup = BeautifulSoup(html, features="html.parser")
    #     results = soup.find_all(re.compile("^el-"))
    #     for tag in results:
    #         if not tag.has_attr("disabled") and not tag.has_attr(":disabled"):
    #             tag[":disabled"] = f"data.{self.widget_id}.disabled"
    #     return str(soup)

    # def wrap(self, html):
    #     return self._wrap_disable_html(html)


class Loading:
    def __init__(self):
        self._loading = False
        if not hasattr(self, "widget_id"):
            self.widget_id = None

    @property
    def loading(self):
        return self._loading

    @loading.setter
    def loading(self, value: bool):
        self._loading = value
        DataJson()[self.widget_id]["loading"] = self._loading
        DataJson().send_changes()

    #
    #  DELETED BECAUSE SHOULD NOT BE USED IN WEBPY RUNTIME
    #
    # def _wrap_loading_html(self, html):
    #     soup = BeautifulSoup(html, features="html.parser")
    #     results = soup.find_all(recursive=False)
    #     for tag in results:
    #         if tag.has_attr("v-loading") or tag.has_attr(":loading"):
    #             return html
    #     for tag in results:
    #         tag["v-loading"] = f"data.{self.widget_id}.loading"
    #     return str(soup)

    # def wrap(self, html):
    #     return self._wrap_loading_html(html)


# def generate_id(cls_name=""):
#     suffix = rand_str(5)  # uuid.uuid4().hex # uuid.uuid4().hex[10]
#     if cls_name == "":
#         return "autoId" + suffix
#     else:
#         return cls_name + "AutoId" + suffix


class Widget(BaseWidget, Hidable, Disableable, Loading):
    def __init__(self, widget_id: str = None):
        # widget_id = self.__init_widget_id(widget_id)
        if widget_id is None:
            raise ValueError("widget_id must be provided for webpy runtime")
        super().__init__(widget_id=widget_id)
        Hidable.__init__(self)
        Disableable.__init__(self)
        Loading.__init__(self)

        self._register()

    #
    # DELETED BECAUSE SHOULD NOT BE USED IN WEBPY RUNTIME
    #
    # def __init_widget_id(self, widget_id: Union[str, None]):
    #     if (
    #         widget_id is not None
    #         and JinjaWidgets().auto_widget_id is True
    #         and ("autoId" in widget_id or "AutoId" in widget_id)
    #     ):
    #         # regenerate id with class name at the beginning
    #         widget_id = generate_id(type(self).__name__)

    #     if widget_id is None:
    #         if JinjaWidgets().auto_widget_id is True:
    #             widget_id = generate_id(type(self).__name__)
    #         else:
    #             try:
    #                 widget_id = varname(frame=2)
    #             except Exception:  # Caller doesn\\\'t assign the result directly to variable(s).
    #                 try:
    #                     widget_id = varname(frame=3)
    #                 except Exception:  # VarnameRetrievingError('Unable to retrieve the ast node.')
    #                     widget_id = generate_id(type(self).__name__)
    #     return widget_id

    def _register(self):
        # get singletons
        data = DataJson()
        # data.raise_for_key(self.widget_id) # TODO
        self.update_data()

        state = StateJson()
        # state.raise_for_key(self.widget_id) # TODO
        self.update_state(state=state)

        #
        # DELETED BECAUSE SHOULD NOT BE USED IN WEBPY RUNTIME
        #
        # JinjaWidgets().context[self.widget_id] = self
        # templates = Jinja2Templates()
        # templates.context_widgets[self.widget_id] = self

    def get_json_data(self):
        raise NotImplementedError()

    def get_json_state(self):
        raise NotImplementedError()

    def update_state(self, state=None):
        if state is None:
            state = StateJson()
        widget_state = self.get_json_state()
        if widget_state is None:
            widget_state = {}
        for cls in Widget.__mro__:
            if "get_json_state" in cls.__dict__ and callable(cls.get_json_state):
                try:
                    cls_state = cls.get_json_state(self)
                except NotImplementedError:
                    continue
                if cls_state is None:
                    continue
                widget_state.update(cls_state)

        state.setdefault(self.widget_id, {}).update(widget_state)

    def update_data(self):
        widget_data = self.get_json_data()
        if widget_data is None:
            widget_data = {}
        for cls in Widget.__mro__:
            if "get_json_data" in cls.__dict__ and callable(cls.get_json_data):
                try:
                    cls_data = cls.get_json_data(self)
                except NotImplementedError:
                    continue
                if cls_data is None:
                    continue
                widget_data.update(cls_data)

        if widget_data is not None:
            DataJson().setdefault(self.widget_id, {}).update(widget_data)

    def get_route_path(self, route: str) -> str:
        return f"/{self.widget_id}/{route}"

    #
    #  DELETED BECAUSE SHOULD NOT BE USED IN WEBPY RUNTIME
    #
    # def to_html(self):
    #     current_dir = Path(self._file_path).parent.absolute()
    #     jinja2_sly_env: Environment = create_env(current_dir)
    #     html = jinja2_sly_env.get_template("template.html").render({"widget": self})
    #     for cls in Widget.__mro__:
    #         if "wrap" in cls.__dict__ and callable(cls.wrap):
    #             html = cls.wrap(self, html)
    #     return markupsafe.Markup(html)

    # def __html__(self):
    #     res = self.to_html()
    #     return res


class ConditionalWidget(Widget):
    def __init__(self, items: List[ConditionalItem], widget_id: str = None):
        self._items = items
        super().__init__(widget_id=widget_id)

    def get_items(self) -> List[ConditionalItem]:
        res = []
        if self._items is not None:
            res.extend(self._items)
        return res


class ConditionalItem:
    def __init__(self, value, label: str = None, content: Widget = None) -> ConditionalItem:
        self.value = value
        self.label = label
        if label is None:
            self.label = str(self.value)
        self.content = content

    def to_json(self):
        return {"label": self.label, "value": self.value}


# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
# https://github.com/pwwang/python-varname
# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string/18425523#18425523
# https://ideone.com/ym3bkD
# https://github.com/pwwang/python-varname
# https://stackoverflow.com/questions/13034496/using-global-variables-between-files
# https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
