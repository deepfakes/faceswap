
class PluginLoader():
    @staticmethod
    def get_extractor(name="Align"):
        module = PluginLoader._import("Extract_{0}".format(name))
        return getattr(module, "Extract")
    
    @staticmethod
    def get_converter(name="Adjust"):
        module = PluginLoader._import("Convert_{0}".format(name))
        return getattr(module, "Convert")
    
    @staticmethod
    def get_model(name="Original"):
        module = PluginLoader._import("Model_{0}".format(name))
        return getattr(module, "Model")
    
    @staticmethod
    def get_trainer(name="Original"):
        module = PluginLoader._import("Model_{0}".format(name))
        return getattr(module, "Trainer")
    
    @staticmethod
    def _import(name):
        return __import__(name, globals(), locals(), [], 1)
