class AttrDict(dict):
    def __init__(self, **kwargs):
        super(AttrDict, self).__init__(**kwargs)
        self.update(kwargs)

    @staticmethod
    def from_dict(dict_v):
        ad = AttrDict()
        ad.update(dict_v)
        return ad

    def __setitem__(self, key: str, value):
        super(AttrDict, self).__setitem__(key, value)
        super(AttrDict, self).__setattr__(key, value)

    def update(self, config: dict, **kwargs):
        for k, v in config.items():
            if k not in self:
                self[k] = AttrDict()
            if isinstance(v, dict):
                self[k].update(v)
            else:
                self[k] = v