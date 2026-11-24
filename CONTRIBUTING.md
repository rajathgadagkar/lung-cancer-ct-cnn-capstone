# Contributing

Contributions, bug fixes, and documentation improvements are welcome!

## Development Setup

```bash
pip install -e ".[ml,dev]"
```

## Quality Checks

Before opening a pull request, please ensure linting and tests pass:

```bash
ruff check src tests
pytest -v
lung-cancer-cnn info --csv data/overview.csv
```

## Guidelines

- Keep modules focused, type-annotated, and well-documented.
- Never commit large raw DICOM datasets or trained model checkpoints (>100 MB) to Git history.
- Maintain compatibility with Python 3.9+ and keep pure-Python CLI/metadata commands working without requiring TensorFlow.

For questions, reach out to **Rajath Gadagkar** at [rajathgadagkar@gmail.com](mailto:rajathgadagkar@gmail.com).
