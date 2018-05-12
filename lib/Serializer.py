#!/usr/bin/env python3
"""
Library providing convenient classes and methods for writing data to files.
"""
import sys
import json
import pickle

try:
    import yaml
except ImportError:
    yaml = None


class Serializer(object):
    ext = ""
    woptions = ""
    roptions = ""

    @classmethod
    def marshal(cls, input_data):
        raise NotImplementedError()

    @classmethod
    def unmarshal(cls, input_string):
        raise NotImplementedError()


class YAMLSerializer(Serializer):
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
    ext = "p"
    woptions = "wb"
    roptions = "rb"

    @classmethod
    def marshal(cls, input_data):
        return pickle.dumps(input_data)

    @classmethod
    def unmarshal(cls, input_bytes):
        return pickle.loads(input_bytes)


def get_serializer(serializer):
    if serializer == "json":
        return JSONSerializer
    elif serializer == "pickle":
        return PickleSerializer
    elif serializer == "yaml" and yaml is not None:
        return YAMLSerializer
    elif serializer == "yaml" and yaml is None:
        print("You must have PyYAML installed to use YAML as the serializer.\n"
              "Switching to JSON as the serializer.", file=sys.stderr)
    return JSONSerializer


def get_serializer_from_ext(ext):
    if ext == ".json":
        return JSONSerializer
    elif ext == ".p":
        return PickleSerializer
    elif ext in (".yaml", ".yml") and yaml is not None:
        return YAMLSerializer
    elif ext in (".yaml", ".yml") and yaml is None:
        print("You must have PyYAML installed to use YAML as the serializer.\n"
              "Switching to JSON as the serializer.", file=sys.stderr)
    return JSONSerializer
