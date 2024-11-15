from huggingface_hub import snapshot_download
import os

mistral_models_path = os.path.join(os.getcwd(), 'mathstral_model')

# Из allow_patterns первые 4 нужны для токенайзера, причем нужно переименовать tokenizer.model.v3 в tokenizer.model
snapshot_download(repo_id="mistralai/Mathstral-7b-v0.1",
                  allow_patterns=["tokenizer_config.json", "tokenizer.model.v3",
                                  "special_tokens_map.json", "tokenizer.json",
                                  "params.json","config.json",
                                  "model-00001-of-00006.safetensors",
                                  "model-00002-of-00006.safetensors",
                                  "model-00003-of-00006.safetensors",
                                  "model-00004-of-00006.safetensors",
                                  "model-00005-of-00006.safetensors",
                                  "model-00006-of-00006.safetensors",
                                  "model.safetensors.index.json"], # ignore_patterns
                  local_dir=mistral_models_path)

