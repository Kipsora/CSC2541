__all__ = [
    'build_config',
    'build_config_from_session_path',
]

import os.path

from flexutils.io import serialize
from flexutils.io.file_system import *
from flexutils.misc import NestedDictMerger, NestedDictifier


def build_config(path: str):
  merger = NestedDictMerger()
  path = get_relative_path(path)

  config = dict()
  if os.path.isfile(path):
    if os.path.basename(path) != "default.yml":
      config['path'], extension = os.path.splitext(path)
      if extension != ".yml":
        raise ValueError("Configuration file should be using .yml "
                         f"as its extension name ({extension} is found)")
      current_config = serialize.load_yaml(path)
      config = merger.merge(current_config, config)
      path = os.path.dirname(path)
    else:
      path = os.path.dirname(path)
      config['path'] = path
  else:
    config['path'] = path

  while not config.get(":root"):
    current_config = serialize.load_yaml(os.path.join(path, "default.yml"))
    config = merger.merge(current_config, config)
    path = os.path.dirname(path)
  config.pop(":root")
  if not config.pop(":leaf", False):
    raise ValueError("The configuration is incomplete (Specify \":leaf: true\" "
                     "to indicate the configuration is complete)")

  return NestedDictifier().dictify(config)


def build_config_from_session_path(session_path: str):
  config = serialize.load_json(os.path.join(session_path, "config.json"))
  config = NestedDictifier().dictify(config)

  path = config.path + ".yml"
  if not os.path.isfile(path):
    path = os.path.join(config.path, "default.yml")
    if not os.path.isfile(path):
      raise ValueError(f"Cannot found the original configuration at "
                       f"\"{simplify_path(path)}\".")

  new_config = build_config(path)
  if config != new_config:
    raise ValueError(f"Configuration was updated and different from "
                     f"{new_config.path}. Please update the session's "
                     f"configuration accordingly.\n")
  return config
