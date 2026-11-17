"""Download the translator bake-off candidates and the FLORES-200 test data into DLSD_MODELS_DIR / data/raw."""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

M = Path(os.environ["DLSD_MODELS_DIR"])
snapshot_download("facebook/nllb-200-3.3B", local_dir=M / "nllb-200-3.3B", allow_patterns=["*.json", "*.model", "*.safetensors", "*.bin"] if False else None,
                  ignore_patterns=["*.msgpack", "*.h5", "*.ot"])
snapshot_download("google/madlad400-10b-mt", local_dir=M / "madlad400-10b-mt", ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.gguf"])
print("done")
