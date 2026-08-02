# Citation

If DynaFX contributes to your research, please cite the software:

```bibtex
@software{Yak_DynaFX,
  author       = {Achref Yakdhane},
  title        = {DynaFX: {A} semantic simulation platform for cognitive digital twins},
  year         = {2026},
  url          = {https://github.com/Achref-Yak/DynaFX},
  version      = {0.2.0},
  license      = {MIT},
}
```

Plain text:

> Achref Yakdhane. (2026). DynaFX: A semantic simulation platform for cognitive
> digital twins (Version 0.2.0). https://github.com/Achref-Yak/DynaFX

The repository also ships a machine-readable [`CITATION.cff`](https://github.com/Achref-Yak/DynaFX/blob/main/CITATION.cff),
so GitHub shows a *Cite this repository* button on the project page.

## Persistent DOI (planned)

A permanent, citable **DOI** for each released version will be minted through
**Zenodo** (CERN's research-data archive). This is not yet active. When we
publish a release, the steps are:

1. **Publish a GitHub Release** with a version tag (e.g. `v0.2.0`).
   Zenodo mints a DOI per release, so each version is citable forever.
2. **Connect the repository on [zenodo.org](https://zenodo.org)** (log in,
   link the GitHub repo, enable the webhook). Zenodo reads `CITATION.cff`
   and `pyproject.toml` metadata automatically.
3. **Add the DOI badge** to the README and to the citation block above.

Until then, cite the version above. If you have an **ORCID**, adding it to
`CITATION.cff` disambiguates the author record; this is optional.
