#! /usr/env/bin/python3
""" Dataclass objects for holding and validating Faceswap Config items """
from __future__ import annotations

import gettext
import logging
from typing import (Any, cast, Generic, get_args, get_origin, get_type_hints,
                    Literal, TypeVar, Union)
import types

from dataclasses import dataclass, field

from lib.utils import get_module_objects


# LOCALES
_LANG = gettext.translation("lib.config", localedir="locales", fallback=True)
_ = _LANG.gettext


logger = logging.getLogger(__name__)
ConfigValueType = bool | int | float | list[str] | str
T = TypeVar("T")


# TODO allow list items other than strings
@dataclass
class ConfigItem(Generic[T]):  # pylint:disable=too-many-instance-attributes
    """ A dataclass for storing config items loaded from config.ini files and dynamically assigning
    and validating that the correct datatype is used.

    The value loaded from the .ini config file can be accessed with either:

    >>> conf.value
    >>> conf()
    >>> conf.get()

    Parameters
    ----------
    datatype : type
        A python type class. This limits the type of data that can be provided in the .ini file
        and ensures that the value is returned to faceswap is correct. Valid datatypes are:
        `int`, `float`, `str`, `bool` or `list`. Note that `list` items must all be strings.
    default : Any
        The default value for this option. It must be of the same type as :attr:`datatype`.
    group : str
        The group that this config item exists within in the config section
    info : str
        A description of what this option does.
    choices : list[str] | Literal["colorchooser"], optional
        If this option's datatype is a `str` then valid selections can be defined here, empty list
        for any value. If the option's datatype is a `list`, then this option must be populated
        with the valid selections. This validates the option and also enables a combobox / radio
        option in the GUI. If the default value is a hex color value, then this should be the
        literal "colorchooser" to present a color choosing interface in the GUI. Ignored for all
        other datatypes
        Default: [] (empty list: no options)
    gui_radio : bool, optional
        If :attr:`choices` are defined, this indicates that the GUI should use radio buttons rather
        than a combobox to display this option. Default: ``False``
    min_max : tuple[int | float, int | float] | None, optional
        For `int` and `float` :attr:`datatype` this is required otherwise it is ignored. Should be
        a tuple of min and max accepted values of the same datatype as the option value. This is
        used for controlling the GUI slider range. Values are not enforced. Default: ``None``
    rounding : int | None, optional
        For `int` and `float :attr:datatypes this is required to be > 0 otherwise it is ignored.
        Used for the GUI slider. For `float`, this is the number of decimal places to display. For
        `int` this is the step size. Default: `-1` (ignored)
    fixed : bool, optional
        [train only]. Training configurations are fixed when the model is created, and then
        reloaded from the state file. Marking an item as fixed=``False`` indicates that this value
        can be changed for existing models, and will override the value saved in the state file
        with the updated value in config. Default: ``True``
    """
    datatype: type[T]
    """ type : A python type class. The datatype of the config value. One of `int`, `float`, `str`,
    `bool` or `list`. `list` will only contain `str` items """
    default: T
    """ Any : The default value for this option. It is of the same type as :attr:`datatype` """
    group: str
    """ str : The group that this config option belongs to """
    info: str
    """ str : A description of what this option does """
    choices: list[str] | Literal["colorchooser"] = field(default_factory=list)
    """ list[str]  | Literal["colorchooser"]: If this option's datatype is a `str` then valid
    selections may be defined here, Empty list if any value is valid. If the datatype is a `list`
    then valid choices will be populated here. If the default value is a hex color code, then the
    literal "colorchooser" will display a color choosing interface in the GUI. """
    gui_radio: bool = False
    """ bool : indicates that the GUI should use radio buttons rather than a combobox to display
    this option if :attr:`choices` is populated """
    min_max: tuple[T, T] | None = None
    """ tuple[int | float, int | float] | None : For `int` and `float` :attr:`datatype` this will
    be populated otherwise it will be ``None``. Used for controlling the GUI slider range. Values
    are not enforced. """
    rounding: int = -1
    """ int : For `int` and `float` :attr:`datatypes` this will be > 0 otherwise it will be `-1`.
    Used for the GUI slider. For `float`, this is the number of decimal places to display. For
    `int` this is the step size. """
    fixed: bool = True
    """ bool : Only used for train.model configurations. Options marked as fixed=``False``
    indicates that this value can be changed for existing models, otherwise the option set when the
    model commenced training is fixed and cannot be changed. Default: ``True`` """
    _value: T = field(init=False)
    """ Any : The value of the config item of type :attr:`datatype`"""
    _name: str = field(init=False)
    """ str: The option name for this object. Set when the config is first loaded """

    @property
    def helptext(self) -> str:
        """ str | Description of the config option with additional formating and helptext added
        from the item parameters """
        retval = f"{self.info}\n"
        if not self.fixed:
            retval += _("\nThis option can be updated for existing models.\n")
        if self.datatype == list:
            retval += _("\nIf selecting multiple options then each option should be separated "
                        "by a space or a comma (e.g. item1, item2, item3)\n")
        if self.choices and self.choices != "colorchooser":
            retval += _("\nChoose from: {}").format(self.choices)
        elif self.datatype == bool:
            retval += _("\nChoose from: True, False")
        elif self.datatype == int:
            assert self.min_max is not None
            cmin, cmax = self.min_max
            retval += _("\nSelect an integer between {} and {}").format(cmin, cmax)
        elif self.datatype == float:
            assert self.min_max is not None
            cmin, cmax = self.min_max
            retval += _("\nSelect a decimal number between {} and {}").format(cmin, cmax)
        default = ", ".join(self.default) if isinstance(self.default, list) else self.default
        retval += _("\n[Default: {}]").format(default)
        return retval

    @property
    def value(self) -> T:
        """ Any : The config value for this item loaded from the config .ini file. String values
        will always be lowercase, regardless of what is loaded from Config """
        retval = self._value
        if isinstance(self._value, str):
            retval = cast(T, self._value.lower())
        if isinstance(self._value, list):
            retval = cast(T, [x.lower() for x in self._value])
        return retval

    @property
    def ini_value(self) -> str:
        """ str : The current value of the ConfigItem as a string for writing to a .ini file """
        if isinstance(self._value, list):
            return ", ".join(str(x) for x in self._value)
        return str(self._value)

    @property
    def name(self) -> str:
        """str: The name associated with this option """
        return self._name

    def _validate_type(self,  # pylint:disable=too-many-return-statements
                       expected_type: Any,
                       attr: Any,
                       depth=1) -> bool:
        """ Validate that provided types are correct when this Dataclass is initialized

        Parameters
        ----------
        expected_type : Any
            The expected data type for the given attribute
        attr : Any
            The attribute to test for correctness
        depth : int, optional
            The current recursion depth

        Returns
        -------
        bool
            ``True`` if the given attribute is a valid datatype

        Raises
        ------
        AssertionError
            On explicit data type failure
        ValueError
            On unhandled data type failure
        """
        value = getattr(self, attr)
        attr_type = type(value)
        expected_type = self.datatype if expected_type == T else expected_type  # type:ignore[misc]

        if attr_type is expected_type:
            return True

        if attr == "datatype":
            assert value in (str, bool, float, int, list), (
                "'datatype' must be one of str, bool, float, int or list. Got {value}")
            return True

        if expected_type == T:  # type:ignore[misc]
            assert attr_type == self.datatype, (
               f"'{attr}' expected: {self.datatype}. Got: {attr_type}")
            return True

        if get_origin(expected_type) is Literal:
            return value in get_args(expected_type)

        if get_origin(expected_type) in (Union, types.UnionType):
            for subtype in get_args(expected_type):
                if self._validate_type(subtype, attr, depth=depth + 1):
                    return True

        if get_origin(expected_type) in (list, tuple) and attr_type in (list, tuple):
            sub_expected = [self.datatype if v == T  # type:ignore[misc]
                            else v for v in get_args(expected_type)]
            return set(type(v) for v in value).issubset(sub_expected)

        if depth == 1:
            raise ValueError(f"'{attr}' expected: {expected_type}. Got: {attr_type}")

        return False

    def _validate_required(self) -> None:
        """ Validate that required parameters are populated

        Raises
        ------
        ValueError
            If any required parameters are empty
        """
        if not self.group:
            raise ValueError("A group must be provided")
        if not self.info:
            raise ValueError("Option info must me provided")

    def _validate_choices(self) -> None:
        """ Validate that choices have been used correctly

        Raises
        ------
        ValueError
            If any choices options have not been populated correctly
        """
        if self.choices == "colorchooser":
            if not isinstance(self.default, str):
                raise ValueError(f"Config Item default must be a string when selecting "
                                 f"choice='colorchooser'. Got {type(self.default)}")
            if not self.default.startswith("#") or len(self.default) != 7:
                raise ValueError(f"Hex color codes should start with a '#' and be 6 "
                                 f"characters long. Got: '{self.default}'")
        elif self.choices and isinstance(self.default, str) and self.default not in self.choices:
            raise ValueError(f"Config item default value '{self.default}' must exist in "
                             f"in choices {self.choices}")

        if isinstance(self.choices, list) and self.choices:
            unique_choices = set(x.lower() for x in self.choices)
            if len(unique_choices) != len(self.choices):
                raise ValueError("Config item choices must be a unique list")
            if isinstance(self.default, list):
                defaults = set(x.lower() for x in self.default)
            else:
                assert isinstance(self.default, str), type(self.default)
                defaults = {self.default.lower()}
            if not defaults.issubset(unique_choices):
                raise ValueError(f"Config item default {self.default} must exist in choices "
                                 f"{self.choices}")

        if not self.choices and isinstance(self.default, list):
            raise ValueError("Config item of type list must have choices defined")

    def _validate_numeric(self) -> None:
        """ Validate that float and int values have been set correctly

        Raises
        ------
        ValueError
            If any float or int options have not been configured correctly
        """
        # NOTE: Have to include datatype filter in next check to exclude bools
        if self.datatype in (float, int) and isinstance(self.default, (float, int)):
            if self.rounding <= 0:
                raise ValueError(f"Config Item rounding must be a positive number for "
                                 f"datatypes float and int. Got {self.rounding}")
            if self.min_max is None or len(self.min_max) != 2:
                raise ValueError(f"Config Item min_max must be a tuple of (<minimum>, "
                                 f"<maximum>) values. Got {self.min_max}")

    def __post_init__(self) -> None:
        """ Validate and type check that the given parameters are valid and set the default value.

        Raises
        ------
        ValueError
            If the Dataclass fails validation checks
        """
        self._name = ""
        self._value = self.default
        try:
            for attr, dtype in get_type_hints(self.__class__).items():
                self._validate_type(dtype, attr)
        except (AssertionError, ValueError) as err:
            raise ValueError(f"Config item failed type checking: {str(err)}") from err

        self._validate_required()
        self._validate_choices()
        self._validate_numeric()

    def get(self) -> T:
        """ Obtain the currently stored configuration value

        Returns
        -------
        Any
            The config value for this item loaded from the config .ini file. String values will
            always be lowecase, regardless of what is loaded from Config """
        return self.value

    def _parse_list(self, value: str | list[str]) -> list[str]:
        """ Parse inbound list values. These can be space/comma-separated strings or a list.

        Parameters
        ----------
        value : str | list[str]
            The inbound value to be converted to a list

        Returns
        -------
        list[str]
            List of strings representing the inbound values.
        """
        if not value:
            return []
        if isinstance(value, list):
            return [str(x) for x in value]
        delimiter = "," if "," in value else None
        retval = list(set(x.strip() for x in value.split(delimiter)))
        logger.debug("[%s] Processed str value '%s' to unique list %s", self._name, value, retval)
        return retval

    def _validate_selection(self, value: str | list[str]) -> str | list[str]:
        """ Validate that the given value is valid within the stored choices

        Parameters
        ----------
        str | list[str]
            The inbound config value to validate

        Returns
        -------
        bool
            ``True`` if the selected value is a valid choice
        """
        assert isinstance(self.choices, list)
        choices = [x.lower() for x in self.choices]
        logger.debug("[%s] Checking config choices", self._name)

        if isinstance(value, str):
            if value.lower() not in choices:
                logger.warning("[%s] '%s' is not a valid config choice. Defaulting to '%s'",
                               self._name, value, self.default)
                return cast(str, self.default)
            return value

        if all(x.lower() in choices for x in value):
            return value

        valid = [x for x in value if x.lower() in choices]
        valid = valid if valid else cast(list[str], self.default)
        invalid = [x for x in value if x.lower() not in choices]
        logger.warning("[%s] The option(s) %s are not valid selections. Setting to: %s",
                       self._name, invalid, valid)

        return valid

    def set(self, value: T) -> None:
        """ Set the item's option value

        Parameters
        ----------
        value : Any
            The value to set this item to. Must be of type :attr:`datatype`

        Raises
        ------
        ValueError
            If the given value does not pass type and content validation checks
        """
        if not self._name:
            raise ValueError("The name of this object should have been set before any value is"
                             "added")

        if self.datatype is list:
            if not isinstance(value, (str, list)):
                raise ValueError(f"[{self._name}] List values should be set as a Str or List. Got "
                                 f"{type(value)} ({value})")
            value = cast(T, self._parse_list(value))

        if not isinstance(value, self.datatype):
            raise ValueError(
                f"[{self._name}] Expected {self.datatype} got {type(value)} ({value})")

        if isinstance(self.choices, list) and self.choices:
            assert isinstance(value, (list, str))
            value = cast(T, self._validate_selection(value))

        if self.choices == "colorchooser":
            assert isinstance(value, str)
            if not value.startswith("#") or len(value) != 7:
                raise ValueError(f"Hex color codes should start with a '#' and be 6 "
                                 f"characters long. Got: '{value}'")

        self._value = value

    def set_name(self, name: str) -> None:
        """ Set the logging name for this object for display purposes

        Parameters
        ----------
        name : str
            The name to assign to this option
        """
        logger.debug("Setting name to '%s'", name)
        assert isinstance(name, str) and name
        self._name = name

    def __call__(self) -> T:
        """ Obtain the currently stored configuration value

        Returns
        -------
        Any
            The config value for this item loaded from the config .ini file. String values will
            always be lowecase, regardless of what is loaded from Config """
        return self.value


@dataclass
class ConfigSection:
    """ Dataclass for holding information about configuration sections and the contained
    configuration items

    Parameters
    ----------
    helptext : str
        The helptext to be displayed for the configuration section
    options : dict[str, :class:`ConfigItem`]
        Dictionary of configuration option name to the options for the section
    """
    helptext: str
    options: dict[str, ConfigItem]


@dataclass
class GlobalSection:
    """ A dataclass for holding and identifying global sub-sections for plugin groups. Any global
    subsections must inherit from this.

    Parameters
    ----------
    helptext : str
        The helptext to be displayed for the global configuration section
    """
    helptext: str


__all__ = get_module_objects(__name__)
