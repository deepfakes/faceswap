
class PluginLoader():
    @staticmethod
    def get_extractor(name):
        return PluginLoader._import("Extract", "Extract_{0}".format(name))
    
    @staticmethod
    def get_converter(name):
        return PluginLoader._import("Convert", "Convert_{0}".format(name))
    
    @staticmethod
    def get_model(name):
        return PluginLoader._import("Model", "Model_{0}".format(name))
    
    @staticmethod
    def get_trainer(name):
        return PluginLoader._import("Trainer", "Model_{0}".format(name))
    
    @staticmethod
    def _import(attr, name):
        print("Loading {} from {} plugin...".format(attr, name))
        module = __import__(name, globals(), locals(), [], 1)
        return getattr(module, attr)
