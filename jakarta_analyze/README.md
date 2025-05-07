# Jakarta Analyze

A Python package for analyzing Jakarta traffic camera footage using computer vision and machine learning techniques.

## Installation

You can install the package directly from source:

```bash
# Install in development mode
pip install -e .
```

## Usage

### As a Command Line Tool

After installation, you can use the `jakarta-analyze` command:

```bash
# Display help
jakarta-analyze --help

# Extract video metadata
jakarta-analyze extract-metadata

# Set logging level
jakarta-analyze extract-metadata --log debug
```

### As a Python Package

You can also import the modules directly in your Python code:

```python
import jakarta_analyze
from jakarta_analyze import get_config, setup, IndentLogger
import logging

# Initialize logging
logger = IndentLogger(logging.getLogger(''), {})
setup("my_script")

# Get configuration
conf = get_config()

# Use the configuration
logger.info(f"Working with configuration: {conf}")
```

## Configuration

The package looks for configuration files in the following order:

1. If `JAKARTAPATH` environment variable is set, it looks in `$JAKARTAPATH/config/`
2. If there's a `config` directory in the current working directory
3. If there's a `config` directory in the package installation directory
4. Otherwise, creates a config directory in `~/.jakarta_analyze/config/`

Configuration files should be in YAML format with `.yml` or `.yaml` extension.

## Converting from Original Project

If you have code that uses the original project structure, update your imports:

From:
```python
from src.modules.utils.setup import setup, IndentLogger
from src.modules.utils.config_loader import get_config
```

To:
```python
from jakarta_analyze.modules.utils.setup import setup, IndentLogger
from jakarta_analyze.modules.utils.config_loader import get_config
```

And remove any references to `JAKARTAPATH` and `PYTHONPATH` environment variables.