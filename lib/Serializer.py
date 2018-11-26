#!/usr/bin/env python3
"""
Library providing convenient classes and methods for writing data to files.
"""
import logging
import json
import pickle

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Serializer():
    """ Parent Serializer class """
    ext = ""
    woptions = ""
    roptions = ""

    @classmethod
    def marshal(cls, input_data):
        """ Override for marshalling """
        raise NotImplementedError()

    @classmethod
    def unmarshal(cls, input_string):
        """ Override for unmarshalling """
        raise NotImplementedError()


class YAMLSerializer(Serializer):
    """ YAML Serializer """
    ext = "yml"
    woptions = "w"
    roptions = "r"

    @classmethod
    def marshal(cls, input_data):
        return yaml.dump(input_data, default_flow_style=False)

    @classmethod
    def unmarshal(cls, input_string):
        return yaml.load(input_string)


class JSONSerializer(Serializer):
    """ JSON Serializer """
    ext = "json"
    woptions = "w"
    roptions = "r"

    @classmethod
    def marshal(cls, input_data):
        return json.dumps(input_data, indent=2)

    @classmethod
    def unmarshal(cls, input_string):
        return json.loads(input_string)


class PickleSerializer(Serializer):
    """ Picke Serializer """
    ext = "p"
    woptions = "wb"
    roptions = "rb"

    @classmethod
    def marshal(cls, input_data):
        return pickle.dumps(input_data)

    @classmethod
    def unmarshal(cls, input_bytes):  # pylint: disable=arguments-differ
        return pickle.loads(input_bytes)


def get_serializer(serializer):
    """ Return requested serializer """
    if serializer == "json":
        return JSONSerializer
    if serializer == "pickle":
        return PickleSerializer
    if serializer == "yaml" and yaml is not None:
        return YAMLSerializer
    if serializer == "yaml" and yaml is None:
        logger.warning("You must have PyYAML installed to use YAML as the serializer."
                       "Switching to JSON as the serializer.")
    return JSONSerializer


def get_serializer_from_ext(ext):
    """ Get the sertializer from filename extension """
    if ext == ".json":
        return JSONSerializer
    if ext == ".p":
        return PickleSerializer
    if ext in (".yaml", ".yml") and yaml is not None:
        return YAMLSerializer
    if ext in (".yaml", ".yml") and yaml is None:
        logger.warning("You must have PyYAML installed to use YAML as the serializer.\n"
                       "Switching to JSON as the serializer.")
    return JSONSerializer
