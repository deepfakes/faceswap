#!/usr/bin python3
""" Pytest unit tests for :mod:`lib.config.ini` """

import os
import pytest

from lib.config import ini as ini_mod

from tests.lib.config.helpers import FakeConfigItem

# pylint:disable=protected-access,invalid-name


_GROUPS = ("group1", "group2", "group3")
_CONFIG = ("custom", "custom_missing", "root", "root_missing")


@pytest.mark.parametrize("plugin_group", _GROUPS)
@pytest.mark.parametrize("config", _CONFIG)
def test_ConfigFile(tmpdir, mocker, plugin_group, config):
    """ Test that :class:`lib.config.ini.ConfigFile` initializes correctly """
    root_conf = tmpdir.mkdir("root").mkdir("config").join(f"{plugin_group}.ini")
    root_dir = os.path.dirname(os.path.dirname(root_conf))
    if config != "root_missing":
        root_conf.write("")
    mocker.patch("lib.config.ini.PROJECT_ROOT", root_dir)

    conf_file = None
    if config.startswith("custom"):
        conf_file = tmpdir.mkdir("config").join("test_custom_config.ini")
        if config == "custom":
            conf_file.write("")

    mock_load = mocker.MagicMock()
    mocker.patch("lib.config.ini.ConfigFile.load", mock_load)

    if config == "custom_missing":  # Error on explicit missing
        with pytest.raises(ValueError):
            ini_mod.ConfigFile("group2test", conf_file)
        return

    instance = ini_mod.ConfigFile(plugin_group, conf_file)
    file_path = conf_file if config == "custom" else root_conf
    assert instance._file_path == file_path
    assert instance._plugin_group == plugin_group
    assert instance._parser.optionxform is str

    if config in ("custom", "root"):  # load when exists
        mock_load.assert_called_once()
    else:
        mock_load.assert_not_called()  # Don't load when it doesn't


def test_ConfigFile_load(mocker):
    """ Test that :class:`lib.config.ini.ConfigFile.load` calls correctly """
    instance = ini_mod.ConfigFile("test")

    mock_read = mocker.MagicMock()
    instance._parser.read = mock_read

    instance.load()

    mock_read.assert_called_once()


def test_ConfigFile_save(mocker):
    """ Test that :class:`lib.config.ini.ConfigFile.save` calls correctly """
    instance = ini_mod.ConfigFile("test")

    mock_write = mocker.MagicMock()
    instance._parser.write = mock_write

    instance.save()

    mock_write.assert_called_once()


class FakeConfigSection:  # pylint:disable=too-few-public-methods
    """ Fake config section """
    def __init__(self, num_opts=2):
        self.options = {f"opt{i}": FakeConfigItem(f"test_value{i}") for i in range(num_opts)}
        self.helptext = f"Test helptext for {num_opts} options"


def get_local_remote(sections=[2, 1, 3]):  # pylint:disable=dangerous-default-value
    """ Obtain an object representing inputs to a ConfigParser and a matching object representing
    Faceswap Config """
    parser_sections = {f"section{i}": {f"opt{idx}": f"test_value{idx}" for idx in range(s)}
                       for i, s in enumerate(sections)}
    fs_sections = {f"section{i}": FakeConfigSection(s) for i, s in enumerate(sections)}
    return parser_sections, fs_sections


def test_ConfigFile_is_synced_structure():
    """ Test that :class:`lib.config.ini.ConfigFile.is_synced_structure` is logical """
    instance = ini_mod.ConfigFile("test")

    sect_sizes = [2, 1, 3]
    parser_sects, fs_sects = get_local_remote(sect_sizes)

    # No Config
    test = instance._is_synced_structure(fs_sects)
    assert test is False

    # Sects exist
    for section in parser_sects:
        instance._parser.add_section(section)

    test = instance._is_synced_structure(fs_sects)
    assert test is False

    # Some Options missing
    for section, options in parser_sects.items():
        for opt, val in options.items():
            instance._parser.set(section, opt, val)
            break

    test = instance._is_synced_structure(fs_sects)
    assert test is False

    # Structure matches
    for section, options in parser_sects.items():
        for opt, val in options.items():
            instance._parser.set(section, opt, val)

    test = instance._is_synced_structure(fs_sects)
    assert test is True

    # Extra saved section
    instance._parser.add_section("text_extra_section")
    test = instance._is_synced_structure(fs_sects)
    assert test is False

    # Structure matches
    del instance._parser["text_extra_section"]
    test = instance._is_synced_structure(fs_sects)
    assert test is True

    # Extra Option
    instance._parser.set(section, "opt_test_extra_option", "val_test_extra_option")
    test = instance._is_synced_structure(fs_sects)
    assert test is False


def testConfigFile_format_help():
    """ Test that :class:`lib.config.ini.ConfigFile.format_help` inserts # on each line """
    instance = ini_mod.ConfigFile("test")
    text = "This\nis a test\n\n\nof some text\n"
    result = instance.format_help(text)
    assert all(x.startswith("#") for x in result.splitlines() if x)


@pytest.mark.parametrize("section",
                         ("section1", "another_section", "section_test"))
def testConfigFile_insert_section(mocker, section):
    """ Test that :class:`lib.config.ini.ConfigFile._insert_section` calls correctly """
    helptext = f"{section}_helptext"

    instance = ini_mod.ConfigFile("test")
    instance.format_help = mocker.MagicMock(return_value=helptext)

    parser = instance._parser

    assert section not in parser

    instance._insert_section(section, helptext, parser)

    instance.format_help.assert_called_once_with(helptext, is_section=True)

    assert section in parser
    assert helptext in parser[section]


@pytest.mark.parametrize(("section", "name", "value"),
                         (("section1", "opt1", "value1"),
                         ("another_section", "my_option", "what_its_worth")))
def testConfigFile_insert_option(mocker, section, name, value):
    """ Test that :class:`lib.config.ini.ConfigFile._insert_option` calls correctly """
    helptext = f"{section}_helptext"

    instance = ini_mod.ConfigFile("test")
    instance.format_help = mocker.MagicMock(return_value=helptext)

    parser = instance._parser
    parser.add_section(section)

    assert name not in parser[section]

    instance._insert_option(section, name, helptext, value, parser)

    instance.format_help.assert_called_once_with(helptext, is_section=False)
    assert name in parser[section]
    assert parser[section][name] == value


_ini, _app,  = get_local_remote([2, 1, 3])
_ini_extra, _app_extra = get_local_remote(sections=[3, 1, 3])
_ini_value, _ = get_local_remote(sections=[2, 1, 3])
_ini_value["section0"]["opt0"] = "updated_value"

_SYNC = ((_ini, _app, "synced"),
         (_ini, _app_extra, "new_from_app"),
         (_ini_extra, _app, "del_from_app"),
         (_ini_value, _app, "updated_ini"))
_SYNC_IDS = [x[-1] for x in _SYNC]


@pytest.mark.parametrize(("ini_config", "app_config", "status"), _SYNC, ids=_SYNC_IDS)
@pytest.mark.parametrize("exists", (True, False), ids=("exists", "not_exists"))
def testConfigFile_sync_from_app(ini_config,  # pylint:disable=too-many-branches  # noqa[C901]
                                 app_config,
                                 status,
                                 exists,
                                 mocker):
    """ Test :class:`lib.config.ini.ConfigFile._sync_from_app` logic """
    mocker.patch("lib.config.ini.ConfigFile._exists", exists)

    instance = ini_mod.ConfigFile("test")
    instance.save = mocker.MagicMock()

    original_parser = instance._parser

    if exists:
        for section, opts in ini_config.items():
            original_parser.add_section(section)
            for name, opt in opts.items():
                original_parser[section][name] = opt

        opt_pairs = [({k: v.value for k, v in opts.options.items()},
                      dict(original_parser[s].items()))
                     for s, opts in app_config.items()]
        # Sanity check that the loaded parser is set correctly
        if status == "synced":
            assert all(set(x[0]) == set(x[1]) for x in opt_pairs)
        elif status == "new_from_app":
            assert any(len(x[1]) < len(x[0]) for x in opt_pairs)
        elif status == "new_from_ini":
            assert any(len(x[0]) < len(x[1]) for x in opt_pairs)
        elif status == "updated_ini":
            vals = [(set(x[0].values()), set(x[1].values())) for x in opt_pairs]
            assert not all(a == i for a, i in vals)
    else:
        for section in ini_config:
            assert section not in instance._parser

    instance._sync_from_app(app_config)  # Sync

    instance.save.assert_called_once()  # Saved
    if exists:
        assert instance._parser is not original_parser  # New config Generated
    else:
        assert instance._parser is original_parser  # Blank Config pre-exists

    opt_pairs = [({k: v.value for k, v in opts.options.items()},
                  {k: v for k, v in instance._parser[s].items() if k.startswith("opt")})
                 for s, opts in app_config.items()]

    # Test options are now in sync
    assert all(set(x[0]) == set(x[1]) for x in opt_pairs)
    # Test that ini value kept
    vals = [(set(x[0].values()), set(x[1].values())) for x in opt_pairs]
    if exists and status == "updated_ini":
        assert any("updated_value" in i for _, i in vals)
        assert any(a != i for a, i in vals)
    else:
        assert not any("updated_value" in i for _, i in vals)
        assert all(a == i for a, i in vals)


@pytest.mark.parametrize(("section", "option", "value", "datatype"),
                         (("section1", "opt_str", "test_str", str),
                          ("section2", "opt_bool", "True", bool),
                          ("section3", "opt_int", "42", int),
                          ("section4", "opt_float", "42.69", float),
                          ("section5", "opt_other", "[test_other]", str)),
                         ids=("str", "bool", "int", "float", "other"))
def testConfigFile_get_converted_value(section, option, value, datatype):
    """ Test :class:`lib.config.ini.ConfigFile._get_converted_value` logic """
    instance = ini_mod.ConfigFile("test")
    instance._parser.add_section(section)
    instance._parser[section][option] = value

    result = instance._get_converted_value(section, option, datatype)
    assert isinstance(result, datatype)
    assert datatype(value) == result


_ini, _app,  = get_local_remote([2, 1, 3])
_ini_changed, _ = get_local_remote(sections=[2, 1, 3])
_ini_changed["section0"]["opt0"] = "updated_value"
_ini_changed["section2"]["opt1"] = "updated_value"

_SYNC_TO = ((_ini, _app, "synced"), (_ini_changed, _app, "updated_ini"))
_SYNC__TO_IDS = [x[-1] for x in _SYNC_TO]


@pytest.mark.parametrize(("ini_config", "app_config", "status"), _SYNC_TO, ids=_SYNC__TO_IDS)
def testConfigFile_sync_to_app(ini_config, app_config, status, mocker):
    """ Test :class:`lib.config.ini.ConfigFile._sync_to_app` logic """

    for sect in app_config.values():  # Add a dummy datatype param to FSConfig
        for opt in sect.options.values():
            setattr(opt, "datatype", str)

    instance = ini_mod.ConfigFile("test")
    instance._get_converted_value = mocker.MagicMock(return_value="updated_value")

    for section, opts in ini_config.items():  # Load up the dummy ini info
        instance._parser.add_section(section)
        for name, opt in opts.items():
            instance._parser[section][name] = opt

    instance._sync_to_app(app_config)

    app_values = {sname: set(v.value for v in sect.options.values())
                  for sname, sect in app_config.items()}
    sect_values = {sname: set(instance._parser[sname].values())
                   for sname in instance._parser.sections()}

    if status == "synced":  # No items change
        instance._get_converted_value.assert_not_called()
    else:  # 2 items updated in the config.ini
        assert instance._get_converted_value.call_count == 2

    # App and ini values must now match
    assert set(app_values) == set(sect_values)
    for sect in app_values:
        assert set(app_values[sect]) == set(sect_values[sect])


@pytest.mark.parametrize("structure_synced",
                         (True, False),
                         ids=("struc_synced", "not_struc_synced"))
@pytest.mark.parametrize("exists", (True, False), ids=("exists", "not_exists"))
def testConfigFile_sync_on_load(structure_synced, exists, mocker):
    """ Test :class:`lib.config.ini.ConfigFile.on_load` logic """
    mocker.patch("lib.config.ini.ConfigFile._exists", exists)
    _, app_config = get_local_remote()

    instance = ini_mod.ConfigFile("test")
    instance._sync_from_app = mocker.MagicMock()
    instance._sync_to_app = mocker.MagicMock()
    instance._is_synced_structure = mocker.MagicMock(return_value=structure_synced)

    instance.on_load(app_config)

    instance._is_synced_structure.assert_called_once_with(app_config)
    instance._sync_to_app.assert_called_once_with(app_config)

    if not exists or not structure_synced:
        instance._sync_from_app.assert_called_with(app_config)
        call_count = 2 if (not exists and not structure_synced) else 1
    else:
        call_count = 0
    assert instance._sync_from_app.call_count == call_count


@pytest.mark.parametrize("app_config",
                         (get_local_remote([2, 1, 3])[1],
                          get_local_remote([4, 2, 6, 8])[1],
                          get_local_remote([3])[1]))
def testConfigFile_sync_update_from_app(app_config, mocker):
    """ Test :class:`lib.config.ini.ConfigFile.update_from_app` logic """
    instance = ini_mod.ConfigFile("test")
    instance.save = mocker.MagicMock()
    for sect in app_config:
        # Updating from app always replaces the existing parser with a new one
        assert sect not in instance._parser.sections()

    instance.update_from_app(app_config)

    instance.save.assert_called_once()
    for sect_name, sect in app_config.items():
        assert sect_name in instance._parser.sections()
        for opt_name, val in sect.options.items():
            assert opt_name in instance._parser[sect_name]
            assert instance._parser[sect_name][opt_name] == val.ini_value
