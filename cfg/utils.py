import collections

from ml_collections import config_dict


def flatten_configdict(cfg: config_dict.ConfigDict, parent_key="", separation_mark="."):
    """Returns a nested OmecaConf dict as a flattened dict, merged with the separation mark."""
    cfg_dict = dict(cfg)
    items = []
    for k, v in cfg_dict.items():
        new_key = parent_key + separation_mark + k if parent_key else k
        if isinstance(v, config_dict.ConfigDict):
            v = dict(v)
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(
                flatten_configdict(
                    cfg=v, parent_key=new_key, separation_mark=separation_mark
                ).items()
            )
        else:
            items.append((new_key, v))
    return dict(items)

