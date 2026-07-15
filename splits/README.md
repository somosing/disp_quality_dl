# Split manifest

The exact industrial scene paths are not distributed in this repository.

After pointing `configs/thesis_final_fullres.yaml` at the prepared 2,470-scene root, run the split command to create a deterministic 2,223/247 split using the configured seed and `validation_count: 247`.

For strict reproduction of the reported run, replace the generated manifest with the archived final manifest if it is available internally.
