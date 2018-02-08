import json, pickle
class Serializer(object):
    ext = "wtf"
    woptions = "????"
    roptions = "????"

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
        try:
            import yaml
        except ImportError:
            print("You must have PyYAML installed to use YAMLSerializer")
        return yaml.dump(input_data, default_flow_style=False)

    @classmethod
    def unmarshal(cls, input_string):
        try:
            import yaml
        except ImportError:
            print("You must have PyYAML installed to use YAMLSerializer")
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
    if serializer == "yaml":
        return YAMLSerializer
    if serializer == "json":
        return JSONSerializer
    if serializer == "pickle":
        return PickleSerializer
    raise NotImplementedError()


def get_serializer_fromext(ext):
    if ext in (".yaml", ".yml"):
        return YAMLSerializer
    if ext == ".json":
        return JSONSerializer
    if ext == ".p":
        return PickleSerializer
    raise NotImplementedError("No serializer matching that file type.")
