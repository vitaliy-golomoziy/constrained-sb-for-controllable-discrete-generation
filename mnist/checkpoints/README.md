# Checkpoint layout

The repository includes exactly the checkpoints used by Section 7:

```text
checkpoints/
├── best.pt
├── mtm_best.pt
└── noise_mtms/
    ├── mtm_t00.pt
    ├── ...
    └── mtm_t08.pt
```

`best.pt` is the VQ-VAE. `mtm_best.pt` is the clean-token masked model.
`noise_mtms/mtm_t00.pt` through `mtm_t08.pt` are the nine independently stored
sampling models used by the Section 7 pipeline. They can be regenerated with
the three training scripts in the parent directory.
