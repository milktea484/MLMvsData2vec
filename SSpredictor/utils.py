from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore

from conf.config import MainConfig, KnotFoldConfig, AdamWConfig

def setup_config():
    """
    OmegaconfのカスタムリゾルバとHydraのConfigStoreへの設定登録を行う関数
    """
    if not OmegaConf._get_resolver("div"):
        OmegaConf.register_new_resolver("div", lambda x, y: int(x / y))

    if not OmegaConf._get_resolver("mul"):
        OmegaConf.register_new_resolver("mul", lambda x, y: x * y)

    if not hasattr(setup_config, "is_registered"):
        cs = ConfigStore.instance()

        cs.store(name="base_config_schema", node=MainConfig)
        
        cs.store(group="model", name="knotfold_schema", node=KnotFoldConfig)
        
        cs.store(group="optimizer", name="adamw_schema", node=AdamWConfig)
        
        setup_config.is_registered = True
