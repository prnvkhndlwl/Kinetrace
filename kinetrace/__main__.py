import os

# No telemetry, before any library is imported: Hugging Face Hub (used by
# transformers for the SAM / ViTPose / RT-DETR downloads) honours both, and
# DO_NOT_TRACK is the cross-tool convention. Kinetrace itself sends nothing.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
# On a Mac the models run on the Apple GPU (Metal / MPS); an operation Metal has
# no kernel for is then computed on the CPU instead of raising. Read by torch at
# import, so it must be set here, before anything imports torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from kinetrace.app import main  # noqa: E402

if __name__ == "__main__":
    main()
