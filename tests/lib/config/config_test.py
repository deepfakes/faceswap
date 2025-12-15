#!/usr/bin python3
""" Pytest unit tests for :mod:`lib.config.config` """

import pytest

from lib.config import config as config_mod

from tests.lib.config.helpers import FakeConfigItem


# pylint:disable=too-few-public-methods,protected-access,invalid-name

def get_instance(mocker, module="plugins.test.test_config"):
    """ Generate a FaceswapConfig instance, substituting the calling module for the one given """
    mocker.patch("lib.config.config.FaceswapConfig.__module__", module)
    return config_mod.FaceswapConfig()


_MODULES = (("plugins.test.test_config", "path_valid"),
            ("plugins.test.test", "path_invalid"),
            ("plugins.config.test_config", "folder_invalid"))
_MODULE_IDS = [x[-1] for x in _MODULES]


@pytest.mark.parametrize(("module", "mod_status"), _MODULES, ids=_MODULE_IDS)
def test_FaceswapConfig_init(module, mod_status, mocker):
    """ Test that :class:`lib.config.config.FaceswapConfig` initializes correctly """
    mocker.patch("lib.config.config.FaceswapConfig.set_defaults", mocker.MagicMock())
    mocker.patch("lib.config.config.ConfigFile.on_load", mocker.MagicMock())
    if mod_status.endswith("invalid"):
        with pytest.raises(AssertionError):
            get_instance(mocker, module=module)
        return

    test = get_instance(mocker, module=module)
    assert test._plugin_group == "test"
    assert isinstance(test._ini, config_mod.ConfigFile)
    test.set_defaults.assert_called_once()
    test._ini.on_load.assert_called_once_with(test.sections)  # pylint:disable=no-member
    assert config_mod._CONFIGS["test"] == test


def test_FaceswapConfig_add_section(mocker):
    """ Test :class:`lib.config.config.FaceswapConfig.add_section` works """
    instance = get_instance(mocker)
    title = "my.test.section"
    info = "And here is some test help text"
    assert title not in instance.sections
    instance.add_section(title, info)
    assert title in instance.sections
    assert isinstance(instance.sections[title], config_mod.ConfigSection)
    assert instance.sections[title].helptext == info


def test_FaceswapConfig_add_item(mocker):
    """ Test :class:`lib.config.config.FaceswapConfig.add_item` works """
    instance = get_instance(mocker)
    section = "my.test.section"
    title = "test_option"
    config_item = "TEST_CONFIG_ITEM"

    assert section not in instance.sections
    with pytest.raises(KeyError):  # Fail adding item to non-existant key
        instance.add_item(section, title, config_item)

    instance.add_section(section, "")
    assert title not in instance.sections[section].options
    instance.add_item(section, title, config_item)
    assert title in instance.sections[section].options
    assert instance.sections[section].options[title] == config_item


@pytest.mark.parametrize("filename",
                         ("test_defaults.py", "train_defaults.py", "different_name.py"))
def test_FaceswapConfig_import_defaults_from_module(mocker, filename):
    """ Test :class:`lib.config.config.FaceswapConfig._defaults_from_module` works """
    mocker.patch("lib.config.config.ConfigItem", FakeConfigItem)

    class DummyMod:
        """ Dummy Module for loading config items """
        opt1 = FakeConfigItem(10)
        opt2 = FakeConfigItem(20)
        invalid = "invalid"
        HELPTEXT = "Test help text"
    mock_mod = mocker.MagicMock(return_value=DummyMod)
    mocker.patch("lib.config.config.import_module", mock_mod)

    instance = get_instance(mocker)
    module_path = "test.module.path"
    plugin_type = "test"
    section = plugin_type + "." + filename[:-3].replace("_defaults", "")

    assert section not in instance.sections

    instance._import_defaults_from_module(filename, module_path, plugin_type)

    mock_mod.assert_called_once_with(f"{module_path}.{filename[:-3]}")

    assert section in instance.sections
    assert instance.sections[section].helptext == DummyMod.HELPTEXT
    assert len(instance.sections[section].options) == 2
    assert isinstance(instance.sections[section].options["opt1"], FakeConfigItem)
    assert isinstance(instance.sections[section].options["opt2"], FakeConfigItem)


def test_FaceswapConfig_defaults_from_plugin(mocker):
    """ Test :class:`lib.config.config.FaceswapConfig._defaults_from_plugin` works """
    mocker.patch("lib.config.config.ConfigItem", FakeConfigItem)
    dir_tree = [("plugins/train/model/plugin_a", [],  ['plugin_a_defaults.py', '__init__.py']),
                ("plugins/extract", [],  ['extract_defaults.py', '__init__.py']),
                ("plugins/convert/writer", [],  ['writer_defaults.py', '__init__.py']),
                ("plugins/train", ["model", "trainer"],  ['train_config.py', '__init__.py'])]
    mock_walk = mocker.MagicMock(return_value=dir_tree)
    mocker.patch("lib.config.config.os.walk", mock_walk)

    instance = get_instance(mocker)

    instance._import_defaults_from_module = mocker.MagicMock()

    instance._defaults_from_plugin("test")

    assert instance._import_defaults_from_module.call_count == 3  # 3 valid, 1 invalid


def test_FaceswapConfig_set_defaults_global(mocker):
    """ Test :class:`lib.config.config.FaceswapConfig.set_defaults` works for global sections """
    mocker.patch("lib.config.config.ConfigItem", FakeConfigItem)

    class DummyMod:
        """ Dummy Module for loading config items """
        opt1 = FakeConfigItem(10)
        opt2 = FakeConfigItem(20)
        invalid = "invalid"
        HELPTEXT = "Test help text"
    mocker.patch("lib.config.config.sys.modules",
                 config_mod.sys.modules | {"plugins.test.test_config": DummyMod})

    instance = get_instance(mocker)

    instance.add_section = mocker.MagicMock()
    instance.add_item = mocker.MagicMock()

    instance.set_defaults("")
    instance.add_section.assert_not_called()
    instance.add_item.assert_not_called()

    instance.set_defaults("test")
    instance.add_section.assert_called_once()
    assert instance.add_item.call_count == 2


def test_FaceswapConfig_set_defaults_subsection(mocker):
    """ Test :class:`lib.config.config.FaceswapConfig.set_defaults` works for sub-sections """
    mocker.patch("lib.config.config.ConfigItem", FakeConfigItem)

    class DummyGlobal(config_mod.GlobalSection):
        """ Dummy GlobalSection class """
        opt1 = FakeConfigItem(30)
        opt2 = FakeConfigItem(40)
        opt3 = FakeConfigItem(50)
        invalid = "invalid"
        helptext = "Section help text"

    class DummyMod:
        """ Dummy Module class for loading config items """
        opt1 = FakeConfigItem(10)
        opt2 = FakeConfigItem(20)
        sect1 = DummyGlobal
        invalid = "invalid"
        HELPTEXT = "Test help text"
    mocker.patch("lib.config.config.sys.modules",
                 config_mod.sys.modules | {"plugins.test.test_config": DummyMod})

    instance = get_instance(mocker)

    instance.add_section = mocker.MagicMock()
    instance.add_item = mocker.MagicMock()

    instance.set_defaults("test")
    assert instance.add_section.call_count == 2  # global + subsection
    assert instance.add_item.call_count == 5  # global + subsection


def test_FaceswapConfig_set_defaults(mocker):
    """ Test :class:`lib.config.config.FaceswapConfig._set_defaults` works """
    instance = get_instance(mocker)

    class DummySection1:
        """ Dummy ConfigSection class """
        options = {"opt1": FakeConfigItem(10),
                   "opt2": FakeConfigItem(20),
                   "opt3": FakeConfigItem(30)}

    class DummySection2:
        """ Dummy ConfigSection class """
        options = {"opt1": FakeConfigItem(40),
                   "opt2": FakeConfigItem(50),
                   "opt3": FakeConfigItem(60)}

    class DummySection3:
        """ Dummy ConfigSection class """
        options = {"opt1": FakeConfigItem(70),
                   "opt2": FakeConfigItem(80),
                   "opt3": FakeConfigItem(90)}

    instance.set_defaults = mocker.MagicMock()
    sections = {"zzz_section": DummySection1(),
                "mmm_section": DummySection2(),
                "aaa_section": DummySection3()}
    instance.sections = sections

    instance._set_defaults()

    instance.set_defaults.assert_called_once()
    for sect_name, sect in instance.sections.items():
        for key, opt in sect.options.items():
            assert opt._name == f"test.{sect_name}.{key}"
    assert list(instance.sections) == sorted(sections)


def test_FaceswapConfig_save(mocker):
    """ Test :class:`lib.config.config.FaceswapConfig.save` works """
    instance = get_instance(mocker)
    instance._ini.update_from_app = mocker.MagicMock()
    instance.sections = "TEST_SECTIONS"

    instance.save_config()

    instance._ini.update_from_app.assert_called_once_with(instance.sections)


def test_get_configs(mocker):
    """ Test :class:`lib.config.config.get_configs` works """
    mock_gen_configs = mocker.MagicMock()
    mocker.patch("lib.config.config.generate_configs", mock_gen_configs)
    mocker.patch("lib.config.config._CONFIGS", "TEST_ALL_CONFIGS")

    result = config_mod.get_configs()
    mock_gen_configs.assert_called_once_with(force=True)
    assert result == "TEST_ALL_CONFIGS"


def test_generate_configs(mocker):
    """ Test :class:`lib.config.config.generate_configs` works """
    _root = "/path/to/faceswap"
    mocker.patch("lib.config.config.PROJECT_ROOT", _root)

    dir_tree = [
        (f"{_root}/plugins/train", [],  ['train_config.py', '__init__.py']),  # Success
        (f"{_root}/plugins/extract", [],  ['extract_config.py', '__init__.py']),  # Success
        (f"{_root}/plugins/convert/writer", [],  ['writer_config.py', '__init__.py']),  # Too deep
        # Wrong name
        (f"{_root}/plugins/train", ["model", "trainer"],  ['train_defaults.py', '__init__.py'])]
    mock_walk = mocker.MagicMock(return_value=dir_tree)
    mocker.patch("lib.config.config.os.walk", mock_walk)

    mock_initialized = mocker.MagicMock()

    class DummyConfig(config_mod.FaceswapConfig):
        """ Dummy FaceswapConfig class """
        def __init__(self,  # pylint:disable=unused-argument,super-init-not-called
                     *args,
                     **kwargs):
            mock_initialized()

    class DummyMod:
        """ Dummy Module to load configs from """
        mod1 = DummyConfig

    mock_mod = mocker.MagicMock(return_value=DummyMod)
    mocker.patch("lib.config.config.import_module", mock_mod)

    config_mod.generate_configs(False)

    assert mock_mod.call_count == 2  # 2 modules imported
    assert mock_initialized.call_count == 2  # 2 configs loaded
