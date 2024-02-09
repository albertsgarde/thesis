import hydra
from hydra.core.config_store import ConfigStore

from . import algorithm
from .algorithm import MASConfig

cs = ConfigStore.instance()

cs.store(name="mas", node=MASConfig)


@hydra.main(config_path="../../conf/mas", version_base="1.3")
def hydra_main(config: MASConfig):
    algorithm.run(config)


if __name__ == "__main__":
    hydra_main()
