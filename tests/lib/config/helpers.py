#! /usr/env/bin/python3
""" Helper mock items for ConfigItems """

import pytest


class FakeConfigItem:
    """ ConfigItem substitute"""
    def __init__(self, value):
        self.value = value
        self._name = ""

    @property
    def ini_value(self):
        """ Dummy ini value """
        return self.value.lower() if isinstance(self.value, str) else self.value

    @property
    def helptext(self):
        """ Dummy help text """
        return f"Test helptext for {self._name}:{self.value}"

    def get(self):
        """ Return the value """
        return self.value

    def set(self, value):
        """ Return the value """
        self.value = value

    def set_name(self, name):
        """ Set the name """
        self._name = name

    def __call__(self):
        return self.value

    def __repr__(self):
        return f"FakeConfigItem(value={self.value!r})"


@pytest.fixture
def patch_config(monkeypatch: pytest.MonkeyPatch):
    """ Fixture to patch user config values """

    def _apply(module, cfg_dict):
        """ Create the fake ConfigItem object """
        for key, value in cfg_dict.items():
            monkeypatch.setattr(module, key, FakeConfigItem(value))

    return _apply
