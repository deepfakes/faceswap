#!/usr/bin/env python3
"""
Library for serializing python objects to and from various different serializer formats
"""
import logging
import json
import os
import pickle

from lib.utils import FaceswapError

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Serializer():
    """ A convenience class for various serializers.

    This class should not be called directly as it acts as the parent for various serializers.
    All serializers should be called from :func:`get_serializer` or
    :func:`get_serializer_from_filename`

    Example
    -------
    >>> from lib.serializer import get_serializer
    >>> serializer = get_serializer('json')
    >>> json_file = '/path/to/json/file.json'
    >>> data = serializer.load(json_file)
    >>> serializer.save(json_file, data)

    """
    def __init__(self):
        self._file_extension = None
        self._write_option = "wb"
        self._read_option = "rb"

    @property
    def file_extension(self):
        """ str: The file extension of the serializer """
        return self._file_extension

    def save(self, filename, data):
        """ Serialize data and save to a file

        Parameters
        ----------
        filename: str
            The path to where the serialized file should be saved
        data: varies
            The data that is to be serialized to file

        Example
        ------
        >>> serializer = get_serializer('json')
        >>> data ['foo', 'bar']
        >>> json_file = '/path/to/json/file.json'
        >>> serializer.save(json_file, data)
        """
        logger.debug("filename: %s, data type: %s", filename, type(data))
        filename = self._check_extension(filename)
        try:
            with open(filename, self._write_option) as s_file:
                s_file.write(self.marshal(data))
        except IOError as err:
            msg = "Error writing to '{}': {}".format(filename, err.strerror)
            raise FaceswapError(msg) from err

    def _check_extension(self, filename):
        """ Check the filename has an extension. If not add the correct one for the serializer """
        extension = os.path.splitext(filename)[1]
        retval = filename if extension else "{}.{}".format(filename, self.file_extension)
        logger.debug("Original filename: '%s', final filename: '%s'", filename, retval)
        return retval

    def load(self, filename):
        """ Load data from an existing serialized file

        Parameters
        ----------
        filename: str
            The path to the serialized file

        Returns
        ----------
        data: varies
            The data in a python object format

        Example
        ------
        >>> serializer = get_serializer('json')
        >>> json_file = '/path/to/json/file.json'
        >>> data = serializer.load(json_file)
        """
        logger.debug("filename: %s", filename)
        try:
            with open(filename, self._read_option) as s_file:
                data = s_file.read()
                logger.debug("stored data type: %s", type(data))
                retval = self.unmarshal(data)
        except IOError as err:
            msg = "Error reading from '{}': {}".format(filename, err.strerror)
            raise FaceswapError(msg) from err
        logger.debug("data type: %s", type(retval))
        return retval

    def marshal(self, data):
        """ Serialize an object

        Parameters
        ----------
        data: varies
            The data that is to be serialized

        Returns
        -------
        data: varies
            The data in a the serialized data format

        Example
        ------
        >>> serializer = get_serializer('json')
        >>> data ['foo', 'bar']
        >>> json_data = serializer.marshal(data)
        """
        logger.debug("data type: %s", type(data))
        try:
            retval = self._marshal(data)
        except Exception as err:
            msg = "Error serializing data for type {}: {}".format(type(data), str(err))
            raise FaceswapError(msg) from err
        logger.debug("returned data type: %s", type(retval))
        return retval

    def unmarshal(self, serialized_data):
        """ Unserialize data to its original object type

        Parameters
        ----------
        serialized_data: varies
            Data in serializer format that is to be unmarshalled to its original object

        Returns
        -------
        data: varies
            The data in a python object format

        Example
        ------
        >>> serializer = get_serializer('json')
        >>> json_data = <json object>
        >>> data = serializer.unmarshal(json_data)
        """
        logger.debug("data type: %s", type(serialized_data))
        try:
            retval = self._unmarshal(serialized_data)
        except Exception as err:
            msg = "Error unserializing data for type {}: {}".format(type(serialized_data),
                                                                    str(err))
            raise FaceswapError(msg) from err
        logger.debug("returned data type: %s", type(retval))
        return retval

    @classmethod
    def _marshal(cls, data):
        """ Override for serializer specific marshalling """
        raise NotImplementedError()

    @classmethod
    def _unmarshal(cls, data):
        """ Override for serializer specific unmarshalling """
        raise NotImplementedError()


class _YAMLSerializer(Serializer):
    """ YAML Serializer """
    def __init__(self):
        super().__init__()
        self._file_extension = "yml"

    @classmethod
    def _marshal(cls, data):
        return yaml.dump(data, default_flow_style=False).encode("utf-8")

    @classmethod
    def _unmarshal(cls, data):
        return yaml.load(data.decode("utf-8"))


class _JSONSerializer(Serializer):
    """ JSON Serializer """
    def __init__(self):
        super().__init__()
        self._file_extension = "json"

    @classmethod
    def _marshal(cls, data):
        return json.dumps(data, indent=2).encode("utf-8")

    @classmethod
    def _unmarshal(cls, data):
        return json.loads(data.decode("utf-8"))


class _PickleSerializer(Serializer):
    """ Pickle Serializer """
    def __init__(self):
        super().__init__()
        self._file_extension = "p"

    @classmethod
    def _marshal(cls, data):
        return pickle.dumps(data)

    @classmethod
    def _unmarshal(cls, data):
        return pickle.loads(data)


def get_serializer(serializer):
    """ Obtain a serializer object

    Parameters
    ----------
    serializer: {'json', 'pickle', yaml'}
        The required serializer format

    Returns
    -------
    serializer: :class:`Serializer`
        A serializer object for handling the requested data format

    Example
    -------
    >>> serializer = get_serializer('json')
    """
    if serializer.lower() == "json":
        return _JSONSerializer()
    if serializer.lower() == "pickle":
        return _PickleSerializer()
    if serializer.lower() == "yaml" and yaml is not None:
        return _YAMLSerializer()
    if serializer.lower() == "yaml" and yaml is None:
        logger.warning("You must have PyYAML installed to use YAML as the serializer."
                       "Switching to JSON as the serializer.")
    logger.warning("Unrecognized serializer: '%s'. Returning json serializer", serializer)
    return _JSONSerializer()


def get_serializer_from_filename(filename):
    """ Obtain a serializer object from a filename

    Parameters
    ----------
    filename: str
        Filename to determine the serializer type from

    Returns
    -------
    serializer: :class:`Serializer`
        A serializer object for handling the requested data format

    Example
    -------
    >>> filename = '/path/to/json/file.json'
    >>> serializer = get_serializer_from_filename(filename)
    """
    logger.debug("filename: '%s'", filename)
    extension = os.path.splitext(filename)[1].lower()
    logger.debug("extension: '%s'", extension)

    if extension == ".json":
        return _JSONSerializer()
    if extension == ".p":
        return _PickleSerializer()
    if extension in (".yaml", ".yml") and yaml is not None:
        return _YAMLSerializer()
    if extension in (".yaml", ".yml") and yaml is None:
        logger.warning("You must have PyYAML installed to use YAML as the serializer.\n"
                       "Switching to JSON as the serializer.")
    logger.warning("Unrecognized extension: '%s'. Returning json serializer", extension)
    return _JSONSerializer()
