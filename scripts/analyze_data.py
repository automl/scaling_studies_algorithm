import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).parent / "data.csv")

allowed_widths = {32, 64, 128, 192, 256}
df = df[df["text_width"].isin(allowed_widths) & df["vision_width"].isin(allowed_widths)]

print(f"Rows after filtering: {len(df)}\n")


def select_min_val_loss(subset: pd.DataFrame, label: str) -> None:
    row = subset.loc[subset["val_loss"].idxmin()]
    print(f"[{label}]")
    print(f"  config_id={row.get('config_id', 'N/A')}  text_width={row['text_width']}  vision_width={row['vision_width']}")
    print(f"  warmup_fraction={row['warmup_fraction']} wd={row['wd']} lr={row['lr']} vision_width={row['vision_width']} text_width={row['text_width']}")
    print(f"  mparams={row['mparams']}  val_loss={row['val_loss']}")
    print(f"  cifar10_mean_per_class_recall={row['cifar10_mean_per_class_recall']}")
    print(f"  imagenet1k_acc1={row['imagenet1k_acc1']}\n")


# From rows with min mparams, pick the one with lowest val_loss
min_mparams_rows = df[df["mparams"] == df["mparams"].min()]
select_min_val_loss(min_mparams_rows, "min mparams → lowest val_loss")

# From rows with min cifar10_mean_per_class_recall, pick the one with lowest val_loss
min_cifar_rows = df[df["cifar10_mean_per_class_recall"] == df["cifar10_mean_per_class_recall"].min()]
select_min_val_loss(min_cifar_rows, "min cifar10_mean_per_class_recall → lowest val_loss")

# From rows with min imagenet1k_acc1, pick the one with lowest val_loss
min_imagenet_rows = df[df["imagenet1k_acc1"] == df["imagenet1k_acc1"].min()]
select_min_val_loss(min_imagenet_rows, "min imagenet1k_acc1 → lowest val_loss")
