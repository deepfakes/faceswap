
class PluginLoader():
    @staticmethod
    def get_extractor(name="Align"):
        return PluginLoader._import("Extract_{0}".format(name), "Extract")
    
    @staticmethod
    def get_converter(name="Adjust"):
        return PluginLoader._import("Convert_{0}".format(name), "Convert")
    
    @staticmethod
    def get_model(name="Original"):
        return PluginLoader._import("Model_{0}".format(name), "Model")
    
    @staticmethod
    def get_trainer(name="Original"):
        return PluginLoader._import("Model_{0}".format(name), "Trainer")
    
    @staticmethod
    def _import(name, attr):
        print("Loading {} from {} plugin...".format(attr, name))
        module = __import__(name, globals(), locals(), [], 1)
        return getattr(module, attr)
