import yaml


class Config:
    __instance = None

    class __Config:
        def __init__(self):
            with open("data/config.yml", 'r') as f:
                self.cfg = yaml.load(f)

    def __init__(self):
        if not Config.__instance:
            Config.__instance = Config.__Config()

    @staticmethod
    def get(key, default=None):
        return Config().__instance.cfg.get(key, default)

