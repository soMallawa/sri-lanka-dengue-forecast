# Security

The released model artifacts are Python joblib pickle files. Pickle loading can execute code.

`dengue-forecast-predict` refuses to load models unless `--trust-pickle` is passed and verifies the model and metadata SHA-256 hashes against `model_manifest_trusted.json` before calling `joblib.load`.

Only load model files from the planned release repository and an immutable Hugging Face commit SHA. The downloader requires a 40-character revision and downloads only trusted manifest paths through Hugging Face `/resolve/{revision}/{path}` URLs. Do not use this loader on arbitrary local directories, mutable revisions, or third-party model files.

The package does not claim that third-party data licenses transfer to model weights. The model artifacts are being released only after separate rights review by the publication owner.
