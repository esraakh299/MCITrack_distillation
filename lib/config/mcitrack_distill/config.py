# Re-export the mcitrack config for distillation training.
# The mcitrack config already includes all distillation parameters.
from lib.config.mcitrack.config import cfg, update_config_from_file
