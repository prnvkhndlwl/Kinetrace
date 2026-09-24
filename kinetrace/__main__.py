import os

# No telemetry, before any library is imported: Hugging Face Hub (used by
# transformers for the SAM / ViTPose / RT-DETR downloads) honours both, and
# DO_NOT_TRACK is the cross-tool convention. Kinetrace itself sends nothing.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

from kinetrace.app import main  # noqa: E402

if __name__ == "__main__":
    main()
