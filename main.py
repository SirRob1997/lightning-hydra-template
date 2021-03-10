
import pytorch_lightning as pl
import hydra
from omegaconf import OmegaConf, DictConfig
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')

@hydra.main(config_path="configs", config_name="defaults")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(1234)
    logger.info(OmegaConf.to_yaml(cfg))

    # Instantiate all modules specified in the configs
    data_module = hydra.utils.instantiate(
        cfg.data,  # Object to instantiate
    )

    model = hydra.utils.instantiate(
        cfg.model,
        # Overwrite arguments at runtime that depends on other modules
        input_dim=cfg.data.input_dim,
        output_dim=cfg.data.output_dim,
        # Don't instantiate optimizer submodules, let `configure_optimizers()` do it
        _recursive_=False,
    )

    # Let hydra manage direcotry output
    tensorboard = pl.loggers.TensorBoardLogger(".", "", "")

    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor='loss/val'),
        pl.callbacks.EarlyStopping(monitor='loss/val', patience=50),
    ]

    trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer),
                         logger=tensorboard,
                         callbacks=callbacks,
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)  # Optionally


if __name__ == '__main__':
    main()
