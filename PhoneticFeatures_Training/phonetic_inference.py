import argparse
from pathlib import Path

import pandas as pd
import torch

try:
    from .phonetic_feature_utils import FEATURE_NAMES, PhonemeFeaturePredictor
except ImportError:
    from phonetic_feature_utils import FEATURE_NAMES, PhonemeFeaturePredictor


CLASS_TO_FEATURE_VALUE = {
    0: -1,
    1: 1,
    2: 0,
}


def load_checkpoint(checkpoint_path, device):
    return torch.load(checkpoint_path, map_location=device)


def resolve_model_config(checkpoint, input_dim=None, hidden_dim=None, num_classes_per_feature=None, active_heads=None):
    saved_config = checkpoint.get("model_config", {})

    resolved_input_dim = input_dim if input_dim is not None else saved_config.get("input_dim")
    resolved_hidden_dim = hidden_dim if hidden_dim is not None else saved_config.get("hidden_dim", 256)
    resolved_num_classes = (
        num_classes_per_feature
        if num_classes_per_feature is not None
        else saved_config.get("num_classes_per_feature")
    )
    resolved_active_heads = active_heads if active_heads is not None else saved_config.get("active_heads")

    missing = []
    if resolved_input_dim is None:
        missing.append("input_dim")
    if resolved_num_classes is None:
        missing.append("num_classes_per_feature")
    if resolved_active_heads is None:
        missing.append("active_heads")

    if missing:
        raise ValueError(
            "Checkpoint does not contain full model_config. "
            f"Provide these arguments manually: {missing}"
        )

    return {
        "input_dim": resolved_input_dim,
        "hidden_dim": resolved_hidden_dim,
        "num_classes_per_feature": resolved_num_classes,
        "active_heads": resolved_active_heads,
    }


def load_checkpoint_model(
    checkpoint_path,
    input_dim=None,
    hidden_dim=None,
    num_classes_per_feature=None,
    active_heads=None,
    device="cpu",
):
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_config = resolve_model_config(
        checkpoint,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes_per_feature=num_classes_per_feature,
        active_heads=active_heads,
    )

    model = PhonemeFeaturePredictor(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_classes_per_feature=model_config["num_classes_per_feature"],
        active_heads=model_config["active_heads"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, model_config


def reduce_logits(logits, reduction="mean"):
    if logits.dim() == 1:
        return logits

    if logits.dim() != 2:
        raise ValueError(f"Expected logits with 1 or 2 dims, got shape {tuple(logits.shape)}")

    if logits.shape[0] == 1:
        return logits.squeeze(0)

    if reduction == "mean":
        return logits.mean(dim=0)
    if reduction == "max":
        return logits.max(dim=0).values
    if reduction == "first":
        return logits[0]

    raise ValueError(f"Unsupported reduction: {reduction}")


def predict_feature_classes(model, embedding, active_heads, reduction="mean"):
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)

    with torch.no_grad():
        outputs = model(embedding)

    predicted_classes = {}
    for head_idx in active_heads:
        logits = reduce_logits(outputs[head_idx], reduction=reduction)
        predicted_classes[FEATURE_NAMES[head_idx]] = int(torch.argmax(logits, dim=0).item())

    return predicted_classes


def decode_feature_classes(predicted_classes):
    return {
        feature_name: CLASS_TO_FEATURE_VALUE[class_idx]
        for feature_name, class_idx in predicted_classes.items()
    }


def predict_feature_classes_sequence(model, embeddings, active_heads):
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)
    if embeddings.dim() != 2:
        raise ValueError(f"Expected embeddings with shape [T, D], got {tuple(embeddings.shape)}")

    with torch.no_grad():
        outputs = model(embeddings)

    result = {}
    for head_idx in active_heads:
        logits = outputs[head_idx]
        result[FEATURE_NAMES[head_idx]] = torch.argmax(logits, dim=1).cpu().tolist()

    return pd.DataFrame(result)


def decode_feature_class_frame(predicted_class_row):
    return {
        feature_name: CLASS_TO_FEATURE_VALUE[int(class_idx)]
        for feature_name, class_idx in predicted_class_row.items()
    }


def decode_feature_classes_sequence(predicted_classes_df):
    decoded_rows = [decode_feature_class_frame(row) for row in predicted_classes_df.to_dict(orient="records")]
    return pd.DataFrame(decoded_rows)


def build_lookup_features(predicted_features, extra_features=None, overrides=None):
    lookup_features = dict(predicted_features)

    if extra_features:
        lookup_features.update(extra_features)

    if overrides:
        lookup_features.update(overrides)

    return lookup_features


def find_matching_ipa_rows(ipa_all_path, find_features):
    df_all = pd.read_csv(ipa_all_path, index_col=0)
    missing_columns = [column for column in find_features if column not in df_all.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in IPA table: {missing_columns}")

    mask = (df_all[list(find_features)] == pd.Series(find_features)).all(axis=1)
    return df_all[mask]


def find_matching_ipa_for_sequence(ipa_all_path, predicted_features_df, extra_features=None, overrides=None):
    df_all = pd.read_csv(ipa_all_path, index_col=0)
    rows = []

    for frame_idx, feature_row in predicted_features_df.iterrows():
        lookup_features = build_lookup_features(
            predicted_features=feature_row.to_dict(),
            extra_features=extra_features,
            overrides=overrides,
        )
        missing_columns = [column for column in lookup_features if column not in df_all.columns]
        if missing_columns:
            raise ValueError(f"Columns not found in IPA table: {missing_columns}")

        mask = (df_all[list(lookup_features)] == pd.Series(lookup_features)).all(axis=1)
        matches = df_all[mask]
        match_symbols = matches.index.tolist()

        rows.append(
            {
                "frame_idx": frame_idx,
                "lookup_features": lookup_features,
                "ipa_matches": match_symbols,
                "num_matches": len(match_symbols),
            }
        )

    return pd.DataFrame(rows)

def find_matching_ipa_for_sequence(ipa_all_path, predicted_features_df, extra_features=None, overrides=None):
    df_all = pd.read_csv(ipa_all_path, index_col=0)
    rows = []

    for frame_idx, feature_row in predicted_features_df.iterrows():


        mask = (df_all[list(feature_row)] == pd.Series(feature_row)).all(axis=1)
        result = df_all[mask]

        rows.append(
            {
                "frame_idx": frame_idx,
                "lookup_features": feature_row,
                "ipa_matches": result,

            }
        )

    return pd.DataFrame(rows)


def predict_ipa_from_embedding(
    embedding,
    checkpoint_path,
    ipa_all_path,
    input_dim=None,
    hidden_dim=None,
    num_classes_per_feature=None,
    active_heads=None,
    device="cpu",
    extra_features=None,
    overrides=None,
    reduction="mean",
):
    model, checkpoint, model_config = load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes_per_feature=num_classes_per_feature,
        active_heads=active_heads,
        device=device,
    )

    predicted_classes = predict_feature_classes(
        model,
        embedding.to(device),
        model_config["active_heads"],
        reduction=reduction,
    )
    predicted_features = decode_feature_classes(predicted_classes)
    lookup_features = build_lookup_features(
        predicted_features=predicted_features,
        extra_features=extra_features,
        overrides=overrides,
    )
    result = find_matching_ipa_rows(ipa_all_path, lookup_features)

    return {
        "predicted_classes": predicted_classes,
        "predicted_features": predicted_features,
        "lookup_features": lookup_features,
        "ipa_matches": result,
        "model_config": model_config,
        "checkpoint_epoch": checkpoint.get("epoch"),
    }


def predict_ipa_for_sequence(
    embeddings,
    checkpoint_path,
    ipa_all_path,
    input_dim=None,
    hidden_dim=None,
    num_classes_per_feature=None,
    active_heads=None,
    device="cpu",
    extra_features=None,
    overrides=None,
):
    model, checkpoint, model_config = load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes_per_feature=num_classes_per_feature,
        active_heads=active_heads,
        device=device,
    )

    predicted_classes_df = predict_feature_classes_sequence(
        model,
        embeddings.to(device),
        model_config["active_heads"],
    )
    predicted_features_df = decode_feature_classes_sequence(predicted_classes_df)
    ipa_matches_df = find_matching_ipa_for_sequence(
        ipa_all_path=ipa_all_path,
        predicted_features_df=predicted_features_df,
        extra_features=extra_features,
        overrides=overrides,
    )

    return {
        "predicted_classes": predicted_classes_df,
        "predicted_features": predicted_features_df,
        "ipa_matches": ipa_matches_df,
        "model_config": model_config,
        "checkpoint_epoch": checkpoint.get("epoch"),
    }


def parse_feature_mapping(raw_values):
    parsed = {}
    for raw_item in raw_values:
        key, value = raw_item.split("=", 1)
        parsed[key] = int(value)
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Infer phonetic features and map them to IPA symbols.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--embedding-path", required=True, help="Path to a torch tensor saved with torch.save.")
    parser.add_argument("--ipa-all-path", required=True)
    parser.add_argument("--input-dim", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--num-classes", nargs="+", type=int, help="Per-feature class counts.")
    parser.add_argument("--active-heads", nargs="+", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--extra-feature", action="append", default=[], help="Fixed feature in key=value form.")
    parser.add_argument("--override", action="append", default=[], help="Override predicted feature in key=value form.")
    parser.add_argument(
        "--reduction",
        choices=["mean", "max", "first"],
        default="mean",
        help="How to collapse multiple embedding rows into one prediction.",
    )
    parser.add_argument(
        "--sequence-mode",
        action="store_true",
        help="Run inference for each embedding row separately instead of collapsing to one prediction.",
    )

    args = parser.parse_args()

    embedding = torch.load(Path(args.embedding_path), map_location=args.device)
    if args.sequence_mode:
        output = predict_ipa_for_sequence(
            embeddings=embedding,
            checkpoint_path=args.checkpoint_path,
            ipa_all_path=args.ipa_all_path,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_classes_per_feature=args.num_classes,
            active_heads=args.active_heads,
            device=args.device,
            extra_features=parse_feature_mapping(args.extra_feature),
            overrides=parse_feature_mapping(args.override),
        )

        print("Predicted classes:")
        print(output["predicted_classes"].head())
        print("\nPredicted panphon-style features:")
        print(output["predicted_features"].head())
        print("\nModel config:")
        print(output["model_config"])
        print("\nIPA matches:")
        print(output["ipa_matches"].head())
    else:
        output = predict_ipa_from_embedding(
            embedding=embedding,
            checkpoint_path=args.checkpoint_path,
            ipa_all_path=args.ipa_all_path,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_classes_per_feature=args.num_classes,
            active_heads=args.active_heads,
            device=args.device,
            extra_features=parse_feature_mapping(args.extra_feature),
            overrides=parse_feature_mapping(args.override),
            reduction=args.reduction,
        )

        print("Predicted classes:")
        print(output["predicted_classes"])
        print("\nPredicted panphon-style features:")
        print(output["predicted_features"])
        print("\nFeatures used for IPA lookup:")
        print(output["lookup_features"])
        print("\nModel config:")
        print(output["model_config"])
        print("\nIPA matches:")
        print(output["ipa_matches"])


if __name__ == "__main__":
    main()
