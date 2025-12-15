#! /usr/env/bin/python
""" Unit tests for lib.convert.objects """
import pytest

from lib.config.objects import ConfigItem
# pylint:disable=invalid-name

_TEST_GROUP = "TestGroup"
_TEST_INFO = "TestInfo"


_STR_CONFIG = (  # type:ignore[var-annotated]
    ("TestDefault", ["TestDefault", "Other"], "success-choices"),
    ("TestDefault", [], "success-no-choices"),
    ("#ffffff", "colorchooser", "success-colorchooser"),
    ("FailDefault", ["TestDefault", "Other"], "fail-choices"),
    ("TestDefault", "Invalid", "fail-invalid-choices"),
    ("TestDefault", "colorchooser", "fail-colorchooser"),
    (1, [], "fail-int"),
    (1.1, [], "fail-float"),
    (True, [], "fail-bool"),
    (["test", "list"], [], "fail-list"))
_STR_PARAMS = ["default", "choices", "status"]


@pytest.mark.parametrize(_STR_PARAMS, _STR_CONFIG, ids=[x[-1] for x in _STR_CONFIG])
def test_ConfigItem_str(default, choices, status):
    """ Test that datatypes validate for strings and value is set correctly """
    dtype = str
    if status.startswith("success"):
        dclass = ConfigItem(datatype=dtype,
                            default=default,
                            group=_TEST_GROUP,
                            info=_TEST_INFO,
                            choices=choices)
        assert dclass.value == default.lower()
    else:
        with pytest.raises(ValueError):
            ConfigItem(datatype=dtype,
                       default=default,
                       group=_TEST_GROUP,
                       info=_TEST_INFO,
                       choices=choices)


_INT_CONFIG = ((10, (0, 100), 1, "success"),
               (20, None, 1, "fail-min-max-missing"),
               (30, (0.1, 100.1), 1, "fail-min-max-dtype"),
               (35, "TestMinMax", 1, "fail-min-max-type"),
               (40, (0, 100), -1, "fail-rounding-missing"),
               (50, (0, 100), 1.1, "fail-rounding-dtype"),
               ("TestDefault", (0, 100), 1, "fail-str"),
               (1.1, (0, 100), 1, "fail-float"),
               (True, (0, 100), 1, "fail-bool"),
               ([0, 1], [0.0, 100.0], 1, "fail-list"))
_INT_PARAMS = ["default", "min_max", "rounding", "status"]


@pytest.mark.parametrize(_INT_PARAMS, _INT_CONFIG, ids=[x[-1] for x in _INT_CONFIG])
def test_ConfigItem_int(default, min_max, rounding, status):
    """ Test that datatypes validate for integers and value is set correctly """
    dtype = int
    if status.startswith("success"):
        dclass = ConfigItem(datatype=dtype,
                            default=default,
                            group=_TEST_GROUP,
                            info=_TEST_INFO,
                            min_max=min_max,
                            rounding=rounding)
        assert dclass.value == default
    else:
        with pytest.raises(ValueError):
            ConfigItem(datatype=dtype,
                       default=default,
                       group=_TEST_GROUP,
                       info=_TEST_INFO,
                       min_max=min_max,
                       rounding=rounding)


_FLOAT_CONFIG = ((10.0, (0.0, 100.0), 1, "success"),
                 (20.1, None, 1, "fail-min-max-missing"),
                 (30.2, (1, 100), 1, "fail-min-max-dtype"),
                 (35.0, "TestMinMax", 1, "fail-min-max-type"),
                 (40.3, (0.0, 100.0), -1, "fail-rounding-missing"),
                 (50.4, (0.0, 100.0), 1.1, "fail-rounding-dtype"),
                 ("TestDefault", (0.0, 100.0), 1, "fail-str"),
                 (1, (0.0, 100.0), 1, "fail-float"),
                 (True, (0.0, 100.0), 1, "fail-bool"),
                 ([0.1, 1.2], [0.0, 100.0], 1, "fail-list"))
_FLOAT_PARAMS = ["default", "min_max", "rounding", "status"]


@pytest.mark.parametrize(_FLOAT_PARAMS, _FLOAT_CONFIG, ids=[x[-1] for x in _FLOAT_CONFIG])
def test_ConfigItem_float(default, min_max, rounding, status):
    """ Test that datatypes validate for floats and value is set correctly """
    dtype = float
    if status.startswith("success"):
        dclass = ConfigItem(datatype=dtype,
                            default=default,
                            group=_TEST_GROUP,
                            info=_TEST_INFO,
                            min_max=min_max,
                            rounding=rounding)
        assert dclass.value == default
    else:
        with pytest.raises(ValueError):
            ConfigItem(datatype=dtype,
                       default=default,
                       group=_TEST_GROUP,
                       info=_TEST_INFO,
                       min_max=min_max,
                       rounding=rounding)


_BOOL_CONFIG = ((True, "success-true"),
                (False, "success-false"),
                ("True", "fail-str"),
                (42, "fail-int"),
                (42.69, "fail-float"),
                ([True, False], "fail-list"))
_BOOL_PARAMS = ["default", "status"]


@pytest.mark.parametrize(_BOOL_PARAMS, _BOOL_CONFIG, ids=[x[-1] for x in _BOOL_CONFIG])
def test_ConfigItem_bool(default, status):
    """ Test that datatypes validate for bool and value is set correctly """
    dtype = bool
    if status.startswith("success"):
        dclass = ConfigItem(datatype=dtype,
                            default=default,
                            group=_TEST_GROUP,
                            info=_TEST_INFO)
        assert dclass.value is default
    else:
        with pytest.raises(ValueError):
            ConfigItem(datatype=dtype,
                       default=default,
                       group=_TEST_GROUP,
                       info=_TEST_INFO)


_LIST_CONFIG = (  # type:ignore[var-annotated]
    (["TestDefault"], ["TestDefault", "Other"], "success"),
    (["TestDefault", "Fail"], ["TestDefault", "Other"], "fail-invalid-choice"),
    (["TestDefault"], [], "fail-no-choices"),
    ([1, 2], [1, 2, 3], "fail-dtype"),
    ("test", ["TestDefault", "Other"], "fail-str"),
    (1, ["TestDefault", "Other"], "fail-int"),
    (1.1, ["TestDefault", "Other"], "fail-float"),
    (True, ["TestDefault", "Other"], "fail-bool"))
_LIST_PARAMS = ["default", "choices", "status"]


@pytest.mark.parametrize(_LIST_PARAMS, _LIST_CONFIG, ids=[x[-1] for x in _LIST_CONFIG])
def test_ConfigItem_list(default, choices, status):
    """ Test that datatypes validate for strings and value is set correctly """
    dtype = list
    if status.startswith("success"):
        dclass = ConfigItem(datatype=dtype,
                            default=default,
                            group=_TEST_GROUP,
                            info=_TEST_INFO,
                            choices=choices)
        assert dclass.value == [x.lower() for x in default]
    else:
        with pytest.raises(ValueError):
            ConfigItem(datatype=dtype,
                       default=default,
                       group=_TEST_GROUP,
                       info=_TEST_INFO,
                       choices=choices)


_REQ_CONFIG = (("TestGroup", "TestInfo", "success"),
               ("", "TestGroup", "fail-no-group"),
               ("TestGroup", "", "fail-no-info"))
_REQ_PARAMS = ["group", "info", "status"]


@pytest.mark.parametrize(_REQ_PARAMS, _REQ_CONFIG, ids=[x[-1] for x in _REQ_CONFIG])
def test_ConfigItem_missing_required(group, info, status):
    """ Test that an error is raised when either group or info are not provided """
    dtype = str
    default = "test"
    if status.startswith("success"):
        dclass = ConfigItem(datatype=dtype,
                            default=default,
                            group=group,
                            info=info)
        assert dclass.group == group
        assert dclass.info == info
        assert isinstance(dclass.helptext, str) and dclass.helptext
        assert dclass.name == ""
    else:
        with pytest.raises(ValueError):
            ConfigItem(datatype=dtype,
                       default=default,
                       group=group,
                       info=info)


_NAME_CONFIG = (("TestName", "success"),
                ("", "fail-no-name"),
                (100, "fail-dtype"))


@pytest.mark.parametrize(("name", "status"), _NAME_CONFIG, ids=[x[-1] for x in _NAME_CONFIG])
def test_ConfigItem_set_name(name, status):
    """ Test that setting the config item's name functions correctly """
    dtype = str
    default = "test"
    dclass = ConfigItem(datatype=dtype,
                        default=default,
                        group="TestGroup",
                        info="TestInfo")
    if status.startswith("success"):
        dclass.set_name(name)
        assert dclass.name == name
    else:
        with pytest.raises(AssertionError):
            dclass.set_name(name)


_STR_SET_CONFIG = (  # type:ignore[var-annotated]
    ("NewValue", ["TestDefault", "NewValue"], "success-choices"),
    ("NoValue", ["TestDefault", "NewValue"], "success-fallback"),
    ("NewValue", [], "success-no-choices"),
    ("#AAAAAA", "colorchooser", "success-colorchooser"),
    ("NewValue", "colorchooser", "fail-colorchooser"),
    (1, [], "fail-int"),
    (1.1, [], "fail-float"),
    (True, [], "fail-bool"),
    (["test", "list"], [], "fail-list"))
_STR_SET_PARAMS = ("value", "choices", "status")


@pytest.mark.parametrize(_STR_SET_PARAMS, _STR_SET_CONFIG, ids=[x[-1] for x in _STR_SET_CONFIG])
def test_ConfigItem_set_str(value, choices, status):
    """ Test that strings validate and set correctly """
    default = "#ffffff" if choices == "colorchooser" else "TestDefault"
    dtype = str
    dclass = ConfigItem(datatype=dtype,
                        default=default,
                        group=_TEST_GROUP,
                        info=_TEST_INFO,
                        choices=choices)

    with pytest.raises(ValueError):  # Confirm setting fails when name not set
        dclass.set(value)

    dclass.set_name("TestName")

    if status.startswith("success"):
        dclass.set(value)
        if status == "success-fallback":
            assert dclass.value == dclass() == dclass.get() == dclass.default.lower()
        else:
            assert dclass.value == dclass() == dclass.get() == value.lower()
    else:
        with pytest.raises(ValueError):
            dclass.set(value)


_INT_SET_CONFIG = ((10, "success"),
                   ("Test", "fail-str"),
                   (1.1, "fail-float"),
                   (["test", "list"], "fail-list"))
_INT_SET_PARAMS = ("value", "status")


@pytest.mark.parametrize(_INT_SET_PARAMS, _INT_SET_CONFIG, ids=[x[-1] for x in _INT_SET_CONFIG])
def test_ConfigItem_set_int(value, status):
    """ Test that ints validate and set correctly """
    default = 20
    dtype = int
    dclass = ConfigItem(datatype=dtype,
                        default=default,
                        group=_TEST_GROUP,
                        info=_TEST_INFO,
                        min_max=(0, 10),
                        rounding=1)

    with pytest.raises(ValueError):  # Confirm setting fails when name not set
        dclass.set(value)

    dclass.set_name("TestName")

    if status.startswith("success"):
        dclass.set(value)
        assert dclass.value == dclass() == dclass.get() == value
    else:
        with pytest.raises(ValueError):
            dclass.set(value)


_FLOAT_SET_CONFIG = ((69.42, "success"),
                     ("Test", "fail-str"),
                     (42, "fail-int"),
                     (True, "fail-bool"),
                     (["test", "list"], "fail-list"))
_FLOAT_SET_PARAMS = ("value", "status")


@pytest.mark.parametrize(_FLOAT_SET_PARAMS,
                         _FLOAT_SET_CONFIG,
                         ids=[x[-1] for x in _FLOAT_SET_CONFIG])
def test_ConfigItem_set_float(value, status):
    """ Test that floats validate and set correctly """
    default = 20.025
    dtype = float
    dclass = ConfigItem(datatype=dtype,
                        default=default,
                        group=_TEST_GROUP,
                        info=_TEST_INFO,
                        min_max=(0.0, 100.0),
                        rounding=1)

    with pytest.raises(ValueError):  # Confirm setting fails when name not set
        dclass.set(value)

    dclass.set_name("TestName")

    if status.startswith("success"):
        dclass.set(value)
        assert dclass.value == dclass() == dclass.get() == value
    else:
        with pytest.raises(ValueError):
            dclass.set(value)


_BOOL_SET_CONFIG = ((True, "success-true"),
                    (False, "success-false"),
                    ("Test", "fail-str"),
                    (42, "fail-int"),
                    (42.69, "fail-float"),
                    (["test", "list"], "fail-list"))
_BOOL_SET_PARAMS = ("value", "status")


@pytest.mark.parametrize(_BOOL_SET_PARAMS, _BOOL_SET_CONFIG, ids=[x[-1] for x in _BOOL_SET_CONFIG])
def test_ConfigItem_set_bool(value, status):
    """ Test that bools validate and set correctly """
    default = True
    dtype = bool
    dclass = ConfigItem(datatype=dtype,
                        default=default,
                        group=_TEST_GROUP,
                        info=_TEST_INFO)

    with pytest.raises(ValueError):  # Confirm setting fails when name not set
        dclass.set(value)

    dclass.set_name("TestName")

    if status.startswith("success"):
        dclass.set(value)
        assert dclass.value == dclass() == dclass.get() == value
    else:
        with pytest.raises(ValueError):
            dclass.set(value)


_LIST_SET_CONFIG = ((["NewValue"], "success-choices"),
                    ("NewValue, TestDefault", "success-delim-comma"),
                    ("NewValue TestDefault", "success-delim-space"),
                    ("NewValue", "success-delim-1value"),
                    (["NoValue"], "success-fallback1"),
                    (["NewValue", "NoValue"], "success-fallback2"),
                    ("NewValue, NoValue", "success-fallback-delim-comma"),
                    ("NewValue NoValue", "success-fallback-delim-space"),
                    ("NoValue", "success-fallback-delim-1value"),
                    (1, "fail-int"),
                    (1.1, "fail-float"),
                    (True, "fail-bool"))
_LIST_SET_PARAMS = ("value", "status")


@pytest.mark.parametrize(_LIST_SET_PARAMS, _LIST_SET_CONFIG, ids=[x[-1] for x in _LIST_SET_CONFIG])
def test_ConfigItem_set_list(value, status):
    """ Test that lists validate and set correctly """
    default = ["TestDefault"]
    choices = ["TestDefault", "NewValue"]
    dtype = list
    dclass = ConfigItem(datatype=dtype,
                        default=default,
                        group=_TEST_GROUP,
                        info=_TEST_INFO,
                        choices=choices)

    with pytest.raises(ValueError):  # Confirm setting fails when name not set
        dclass.set(value)

    dclass.set_name("TestName")

    if status.startswith("success"):
        dclass.set(value)

        if not isinstance(value, list):
            value = [x.strip() for x in value.split(",")] if "," in value else value.split()
        assert dclass.value == dclass() == dclass.get()
        expected = [x.lower() for x in value]
        if status.startswith("success-fallback"):
            expected = [x.lower() for x in value if x in choices]
            if not expected:
                expected = [x.lower() for x in default]
        assert set(expected) == set(dclass.value)

    else:
        with pytest.raises(ValueError):
            dclass.set(value)
