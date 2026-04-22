from typing import Any, Dict, List, Optional, Tuple
import hydra
import torch
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf.dictconfig import DictConfig

from matcha import utils

torch.serialization.add_safe_globals([DictConfig])
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #


log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")  # pylint: disable=protected-access
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")  # pylint: disable=protected-access
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")  # pylint: disable=protected-access
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        if cfg.get("ckpt_path"):
            log.info(f"💉 Injecting pre-trained VCTK weights from: {cfg.ckpt_path}")
            checkpoint = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint["state_dict"]
            model_dict = model.state_dict()

            new_state_dict = {}
            for k, v in state_dict.items():
                if k in model_dict:
                    # If shapes match perfectly, copy them directly
                    if v.shape == model_dict[k].shape:
                        new_state_dict[k] = v
                        
                    # If the Vocab size changed (178 -> 198)
                    elif "encoder.emb.weight" in k:
                        log.info(f"✂️ Grafting Vocab: {v.shape} -> {model_dict[k].shape}")
                        new_w = model_dict[k].clone() # Keep new random weights

                        min_v = min(v.shape[0], model_dict[k].shape[0])
                        min_d = min(v.shape[1], model_dict[k].shape[1])

                        new_w[:min_v, :min_d] = v[:min_v, :min_d] # Inject old weights
                        new_state_dict[k] = new_w
                        
                    # If the Speaker size changed (Accent Vector Setup)
                    elif "spk_emb.weight" in k:
                        log.info(f"✂️ Grafting VCTK Anchors (0 Female, 2 Male) to 14 Accent Speakers")
                        new_w = model_dict[k].clone()
                        min_d = min(v.shape[1], model_dict[k].shape[1])

                        vctk_female_anchor = 0  
                        vctk_male_anchor = 2    

                        # 🚨 UPDATE THESE LISTS to match your exact metadata IDs!
                        # ID 0 is our Female Anchor, ID 1 is our Male Anchor
                        # Sort your remaining 12 L2-ARCTIC IDs by gender below:
                        female_speaker_ids = [0, 2, 4, 6, 8, 10, 12]
                        male_speaker_ids = [1, 3, 5, 7, 9, 11, 13]

                        # Graft all females to the 0th VCTK speaker
                        for spk_id in female_speaker_ids:
                            if spk_id < new_w.shape[0]:
                                new_w[spk_id, :min_d] = v[vctk_female_anchor, :min_d]
                                
                        # Graft all males to the 2nd VCTK speaker
                        for spk_id in male_speaker_ids:
                            if spk_id < new_w.shape[0]:
                                new_w[spk_id, :min_d] = v[vctk_male_anchor, :min_d]

                        new_state_dict[k] = new_w
                        
                    else:
                        log.info(f"⚠️ Skipping {k} due to unhandled size mismatch: {v.shape} vs {model_dict[k].shape}")

            # strict=False handles any completely missing layers
            model.load_state_dict(new_state_dict, strict=False)
            log.info("✅ Pre-trained weights successfully loaded!")
            
        log.info("🚀 Launching Trainer.fit from Epoch 0...")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=None, weights_only=False)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        #trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        print("🎉 Training complete! Skipping formal test_step as it is not defined.")
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter