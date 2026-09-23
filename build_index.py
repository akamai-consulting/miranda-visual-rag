from pathlib import Path
import argparse
import random
import time

import torch
import torch.nn.functional as F
import open_clip
from PIL import Image


# ============================================================
# COMMAND LINE
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Build OpenCLIP image indexes for Miranda. "
        "Multiple image roots may be supplied."
    )
)

parser.add_argument(
    "--image-root",
    action="append",
    required=True,
    help=(
        "Directory containing images. "
        "May be supplied multiple times."
    ),
)

parser.add_argument(
    "--output-dir",
    default="indexes",
    help=(
        "Directory in which generated index files are stored. "
        "Default: indexes"
    ),
)

args = parser.parse_args()


IMAGE_ROOTS = [
    Path(path).expanduser().resolve()
    for path in args.image_root
]

OUTPUT_DIR = Path(
    args.output_dir
).expanduser().resolve()


# ============================================================
# CONFIG
# ============================================================

INDEX_SIZES = [
    500,
    5_000,
    50_000,
    200_000,
    289_222,
]

BATCH_SIZE = 32

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

SEED = 42


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print(f"Using device: {device}")


# ============================================================
# DEVICE SYNC
# ============================================================

def sync_device():

    if device.type == "mps":

        torch.mps.synchronize()

    elif device.type == "cuda":

        torch.cuda.synchronize()


# ============================================================
# FIND IMAGES
# ============================================================

image_paths = []


print()
print("Image sources:")


for image_root in IMAGE_ROOTS:

    if not image_root.exists():

        raise RuntimeError(
            f"Image root does not exist: {image_root}"
        )

    if not image_root.is_dir():

        raise RuntimeError(
            f"Image root is not a directory: {image_root}"
        )


    root_images = sorted(
        list(image_root.rglob("*.jpg"))
        + list(image_root.rglob("*.jpeg"))
        + list(image_root.rglob("*.JPG"))
        + list(image_root.rglob("*.JPEG"))
    )


    print(
        f"  {image_root}"
    )

    print(
        f"    {len(root_images):,} images"
    )


    image_paths.extend(
        root_images
    )


# ------------------------------------------------------------
# Remove duplicates.
#
# This protects against accidentally supplying overlapping
# image roots.
# ------------------------------------------------------------

image_paths = sorted(
    set(image_paths)
)


print()
print(
    f"Found {len(image_paths):,} unique images."
)


if not image_paths:

    raise RuntimeError(
        "No images found under supplied image roots."
    )


# ============================================================
# CREATE DETERMINISTIC MASTER SAMPLE
# ============================================================

largest_requested = len(
    image_paths
)

largest_size = min(
    largest_requested,
    len(image_paths)
)


random.seed(
    SEED
)


master_sample = random.sample(
    image_paths,
    largest_size
)


print(
    f"Master sample: {len(master_sample):,}"
)


# ============================================================
# LOAD OPENCLIP
# ============================================================

print()
print("Loading OpenCLIP...")


model, _, preprocess = (
    open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED
    )
)


model = model.to(
    device
)

model.eval()


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print(
    f"Output directory: {OUTPUT_DIR}"
)


# ============================================================
# EMBED MASTER SAMPLE ONCE
# ============================================================

all_embeddings = []
valid_paths = []


start_total = time.perf_counter()


for batch_start in range(
    0,
    len(master_sample),
    BATCH_SIZE
):

    batch_paths = master_sample[
        batch_start:
        batch_start + BATCH_SIZE
    ]


    tensors = []
    successful_paths = []


    for path in batch_paths:

        try:

            with Image.open(path) as img:

                image = img.convert(
                    "RGB"
                )

                tensors.append(
                    preprocess(image)
                )


            successful_paths.append(
                str(path)
            )


        except Exception as exc:

            print(
                f"Skipping {path}: {exc}"
            )


    if not tensors:

        continue


    batch = torch.stack(
        tensors
    ).to(
        device
    )


    sync_device()


    with torch.no_grad():

        embeddings = (
            model.encode_image(
                batch
            )
        )


        embeddings = F.normalize(
            embeddings,
            dim=-1
        )


    sync_device()


    all_embeddings.append(
        embeddings.cpu()
    )


    valid_paths.extend(
        successful_paths
    )


    completed = min(
        batch_start + BATCH_SIZE,
        len(master_sample)
    )


    if (
        completed % 1000 == 0
        or completed == len(master_sample)
    ):

        elapsed = (
            time.perf_counter()
            - start_total
        )


        rate = (
            completed / elapsed
        )


        print(
            f"{completed:,}/"
            f"{len(master_sample):,} "
            f"| {rate:.1f} images/sec"
        )


# ============================================================
# FINAL MASTER MATRIX
# ============================================================

if not all_embeddings:

    raise RuntimeError(
        "No images were successfully embedded."
    )


embedding_matrix = torch.cat(
    all_embeddings,
    dim=0
)


elapsed_total = (
    time.perf_counter()
    - start_total
)


print()
print("Embedding complete.")


print(
    "Matrix shape:",
    embedding_matrix.shape
)


print(
    f"Valid images: "
    f"{len(valid_paths):,}"
)


print(
    f"Total time: "
    f"{elapsed_total:.2f} sec"
)


print(
    f"Average: "
    f"{elapsed_total / len(valid_paths) * 1000:.2f} "
    f"ms/image"
)


# ============================================================
# SAVE MULTIPLE NESTED INDEXES
# ============================================================

print()
print("Saving indexes...")


for requested_size in INDEX_SIZES:

    actual_size = min(
        requested_size,
        len(valid_paths)
    )


    output_file = (
        OUTPUT_DIR
        / f"deepfashion_index_{actual_size}.pt"
    )


    subset_embeddings = (
        embedding_matrix[
            :actual_size
        ]
        .clone()
    )


    subset_paths = (
        valid_paths[
            :actual_size
        ]
    )


    torch.save(
        {
            "embeddings": subset_embeddings,

            "paths": subset_paths,

            "catalog_size": actual_size,

            "model_name": MODEL_NAME,

            "pretrained": PRETRAINED,

            "seed": SEED,

            "image_roots": [
                str(path)
                for path in IMAGE_ROOTS
            ],
        },
        output_file
    )


    file_mb = (
        output_file.stat().st_size
        / 1024
        / 1024
    )


    print(
        f"{actual_size:>8,} vectors "
        f"→ {output_file} "
        f"({file_mb:.1f} MiB)"
    )


print()
print("Done.")