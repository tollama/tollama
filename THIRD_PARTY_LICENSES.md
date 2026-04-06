# Third-Party Licenses

This file lists the licenses of tollama's Python dependencies.
Generated with [pip-licenses](https://pypi.org/project/pip-licenses/).

> **Note:** This inventory covers the `[dev]` install profile.
> Runner-family extras (`runner_torch`, `runner_timesfm`, etc.) pull additional
> dependencies; regenerate when shipping wheels or containers that include them.

| Name | Version | License | Author |
|------|---------|---------|--------|
| PyYAML | 6.0.3 | MIT License | Kirill Simonov |
| annotated-doc | 0.0.4 | MIT | Sebastián Ramírez |
| annotated-types | 0.7.0 | MIT License | Adrian Garcia Badaracco, Samuel Colvin, Zac Hatfield-Dodds |
| anyio | 4.12.1 | MIT | Alex Grönholm |
| certifi | 2026.1.4 | Mozilla Public License 2.0 (MPL 2.0) | Kenneth Reitz |
| click | 8.3.1 | BSD-3-Clause | Pallets |
| fastapi | 0.129.0 | MIT | Sebastián Ramírez |
| filelock | 3.24.3 | MIT | UNKNOWN |
| fsspec | 2026.2.0 | BSD-3-Clause | UNKNOWN |
| h11 | 0.16.0 | MIT License | Nathaniel J. Smith |
| hf-xet | 1.3.1 | Apache-2.0 | UNKNOWN |
| httpcore | 1.0.9 | BSD-3-Clause | Tom Christie |
| httpx | 0.28.1 | BSD License | Tom Christie |
| huggingface_hub | 0.36.2 | Apache Software License | Hugging Face, Inc. |
| idna | 3.11 | BSD-3-Clause | Kim Davies |
| iniconfig | 2.3.0 | MIT | Ronny Pfannschmidt, Holger Krekel |
| markdown-it-py | 4.0.0 | MIT License | Chris Sewell |
| mdurl | 0.1.2 | MIT License | Taneli Hukkinen |
| numpy | 2.4.2 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Travis E. Oliphant et al. |
| packaging | 26.0 | Apache-2.0 OR BSD-2-Clause | Donald Stufft |
| pandas | 2.3.3 | BSD License | The Pandas Development Team |
| pluggy | 1.6.0 | MIT License | Holger Krekel |
| pydantic | 2.12.5 | MIT | Samuel Colvin et al. |
| pydantic_core | 2.41.5 | MIT | Samuel Colvin et al. |
| pytest | 9.0.2 | MIT | Holger Krekel, Bruno Oliveira et al. |
| pytest-asyncio | 1.3.0 | Apache-2.0 | Tin Tvrtković |
| python-dateutil | 2.9.0 | Apache Software License; BSD License | Gustavo Niemeyer |
| python-multipart | 0.0.22 | Apache-2.0 | Andrew Dunham, Marcelo Trylesinski |
| pytz | 2025.2 | MIT License | Stuart Bishop |
| rich | 14.3.3 | MIT License | Will McGugan |
| ruff | 0.15.3 | MIT License | Astral Software Inc. |
| shellingham | 1.5.4 | ISC License (ISCL) | Tzu-ping Chung |
| six | 1.17.0 | MIT License | Benjamin Peterson |
| sniffio | 1.3.1 | MIT OR Apache-2.0 | Nathaniel J. Smith |
| starlette | 0.52.1 | BSD-3-Clause | Tom Christie |
| tqdm | 4.67.3 | MPL-2.0 AND MIT | UNKNOWN |
| typer | 0.24.1 | MIT | Sebastián Ramírez |
| typing-inspection | 0.4.2 | MIT | Victorien Plot |
| typing_extensions | 4.15.0 | PSF-2.0 | Guido van Rossum et al. |
| tzdata | 2025.3 | Apache-2.0 | Python Software Foundation |
| urllib3 | 2.6.3 | MIT | Andrey Petrov |
| uvicorn | 0.41.0 | BSD-3-Clause | Tom Christie |

## Non-PyPI Dependencies

### TimesFM

- **Source:** `https://github.com/google-research/timesfm/archive/2dcc66fbfe2155adba1af66aa4d564a0ee52f61e.tar.gz`
- **License:** Apache-2.0 (verified at pinned commit)
- **Note:** Installed only via the optional `runner_timesfm` extra.
