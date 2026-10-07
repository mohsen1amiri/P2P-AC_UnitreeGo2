# Vendored PyYAML

The pure-Python part of **PyYAML 6.0.2** (`lib/yaml/` of the official source release on PyPI,
sha256 of `PyYAML-6.0.2.tar.gz`: `d584d9ec91ad65861cc08d42e834324ef890a082e591037abe114850ff7bbc3e`), unmodified.
MIT license (`LICENSE`).

It is used only when the Python that runs `scripts/make_sweep.py` or the analysis has no `yaml` package, for example
the system `python3` of an HPC login node, so these tools need nothing installed. Inside the container the regular
PyYAML is used.
