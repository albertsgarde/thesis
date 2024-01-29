from dataclasses import dataclass

import hydra
import sparse_autoencoder as sae  # type: ignore[reportMissingTypeStubs, import-untyped]
from hydra.core.config_store import ConfigStore
from sparse_autoencoder import (  # type: ignore[reportMissingTypeStubs]
    ActivationResamplerHyperparameters,
    AutoencoderHyperparameters,
    Hyperparameters,
    LossHyperparameters,
    Method,
    OptimizerHyperparameters,
    Parameter,
    PipelineHyperparameters,
    SourceDataHyperparameters,
    SourceModelHyperparameters,
    SweepConfig,
)
from transformer_lens.HookedTransformer import HookedTransformer


@dataclass
class SAEConfig:
    model_name: str = "solu-1l"
    expansion_factor: int = 4


def main(config: SAEConfig):
    model = HookedTransformer.from_pretrained(config.model_name, device="cpu")  # type: ignore[reportUnknownVariableType]
    sweep_config = SweepConfig(
        parameters=Hyperparameters(
            loss=LossHyperparameters(
                l1_coefficient=Parameter(max=0.03, min=0.008),
            ),
            optimizer=OptimizerHyperparameters(
                lr=Parameter(max=0.001, min=0.00001),
            ),
            source_model=SourceModelHyperparameters(
                name=Parameter("gpt2"),
                cache_names=Parameter([f"blocks.{layer}.hook_mlp_out" for layer in range(1)]),
                hook_dimension=Parameter(768),
            ),
            source_data=SourceDataHyperparameters(
                dataset_path=Parameter("alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2"),
                context_size=Parameter(256),
                pre_tokenized=Parameter(value=True),
                pre_download=Parameter(value=False),  # Default to streaming the dataset
            ),
            autoencoder=AutoencoderHyperparameters(expansion_factor=Parameter(value=1)),
            pipeline=PipelineHyperparameters(
                max_activations=Parameter(1_000),
                checkpoint_frequency=Parameter(1_000),
                validation_frequency=Parameter(1_000),
                max_store_size=Parameter(820_000),
            ),
            activation_resampler=ActivationResamplerHyperparameters(
                resample_interval=Parameter(10_000),
                n_activations_activity_collate=Parameter(10_000),
                threshold_is_dead_portion_fires=Parameter(1e-6),
                max_n_resamples=Parameter(4),
            ),
        ),
        method=Method.RANDOM,
    )

    # print(model.W_in.shape)
    sae.sweep(sweep_config)


cs = ConfigStore.instance()

cs.store(name="sae", node=SAEConfig)


@hydra.main(config_path="../conf/sae", version_base="1.3")
def hydra_main(config: SAEConfig):
    main(config)


if __name__ == "__main__":
    hydra_main()
