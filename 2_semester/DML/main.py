import argparse

import pytorch_lightning as pl

from data import get_dataloader
from models import EmoClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--feature_type", type=str, choices=["tf_idf", "w2v", "mel", "mfcc", "conformer", "bert", "early_fusion", "early_fusion_v2", "late_fusion", "cross-attn-fusion"], required=True)
    parser.add_argument("--mosei_path", type=str, required=True, help="Path to mosei folder")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--transfer_learning", type=bool, default=True)
    return parser.parse_args()


def main(args) -> None:
    pl.seed_everything(42, verbose=False)
    lm = EmoClassifier(args.feature_type, num_classes=3, transfer_learning=args.transfer_learning, num_layers=args.num_layers, num_heads=args.num_heads)
    metric = "val_f1"
    mode = "max"
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor=metric,
            mode=mode,
            patience=2,
            min_delta=0.02,
            verbose=False,
        ),
        pl.callbacks.ModelCheckpoint(
            monitor=metric,
            mode=mode,
            dirpath="checkpoints",
            filename=f"ckpt_{args.feature_type}",
            auto_insert_metric_name=False,
            save_top_k=1,
        ),
    ]
    logger = pl.loggers.CSVLogger(
        "logs",
        name=f"{args.feature_type}",
    )
    trainer = pl.Trainer(
        max_epochs=15,
        num_sanity_val_steps=0,
        enable_model_summary=False,
        gradient_clip_val=1,
        deterministic=True,
        benchmark=False,
        callbacks=callbacks,
        logger=[logger],
    )
    ckpt_path = args.ckpt_path
    if args.train:
        train_loader = get_dataloader(args.feature_type, "train", args.mosei_path)
        val_loader = get_dataloader(args.feature_type, "val", args.mosei_path)
        trainer.fit(lm, train_dataloaders=train_loader, val_dataloaders=val_loader)
        ckpt_path = "best"
    if args.test:
        if ckpt_path is None:
            raise RuntimeError("No checkpoint path was passed")
        test_loader = get_dataloader(args.feature_type, "test")
        trainer.test(lm, dataloaders=test_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main(parse_args())
