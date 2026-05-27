# Atria Hub

A Python client library for interacting with the Atriax Hub API, providing seamless access to datasets, models, and authentication services.

## 🚀 Features

- **Authentication Management**: Secure authentication with credential storage using Supabase
- **Dataset Operations**: Upload, download, and manage datasets with versioning support
- **Model Management**: Access and manage machine learning models
- **File System Integration**: Built-in support for LakeFS file operations
- **Health Monitoring**: Health check utilities for service monitoring
- **Repository Management**: Automated credential management for repository access

## 📦 Installation

```bash
pip install atria_cli
```

### Development Installation

For development, clone the repository and install with development dependencies:

```bash
git clone https://github.com/saifullah3396/atria_cli.git
cd atria_cli
uv sync --dev
```

## 🔧 Requirements

- Python 3.11
- Dependencies managed via `uv` or `pip`

## 🚀 Quick Start

### Basic Usage

```python
from atria_cli import AtriaHub

# Initialize the hub client
hub = AtriaHub()

# Access datasets
dataset = hub.datasets.get("my-dataset")

# Upload files to a dataset
hub.datasets.upload_files(
    dataset=dataset,
    branch="main",
    dataset_files=[("local/path/file.txt", "remote/path/file.txt")]
)

# Download dataset files
hub.datasets.download_files(
    dataset=dataset,
    branch="main",
    destination_path="./downloads/"
)
```

### Authentication

```python
from atria_cli import AtriaHub
from atria_cli.models import AuthLoginModel

# Initialize with custom credentials
credentials = AuthLoginModel(email="your@email.com", password="password")
hub = AtriaHub(credentials=credentials)

# Force sign-in (useful for credential refresh)
hub = AtriaHub(force_sign_in=True)
```

### Custom Configuration

```python
from atria_cli import AtriaHub

# Initialize with custom base URL and API key
hub = AtriaHub(
    base_url="https://your-custom-hub.com",
    anon_api_key="your-api-key",
    service_name="custom-service"
)
```

## 📚 API Reference

### AtriaHub Class

The main entry point for interacting with the Atria Hub services.

**Properties:**
- `client`: Access to the underlying HTTP client
- `auth`: Authentication API
- `datasets`: Dataset management API
- `models`: Model management API
- `health_check`: Health monitoring API

### Datasets API

**Methods:**
- `get(name)`: Retrieve a dataset by name
- `create(body)`: Create a new dataset
- `get_or_create(name, description, data_instance_type, is_public)`: Get existing or create new dataset
- `upload_files(dataset, branch, dataset_files)`: Upload files to a dataset
- `download_files(dataset, branch, destination_path)`: Download dataset files

### Models API

Access and manage machine learning models through the models API.

### Authentication API

Handle user authentication and session management with secure credential storage.

## 🛠️ Development

### Scripts

The project includes several utility scripts in the `scripts/` directory:

- `build_client.sh`: Build the client
- `format.sh`: Format code using Ruff
- `lint.sh`: Lint code using Ruff
- `test.sh`: Run tests

### Code Quality

The project uses:
- **Ruff**: For linting and formatting
- **MyPy**: For type checking
- **Pytest**: For testing
- **Coverage**: For test coverage reporting

### Running Tests

```bash
# Run tests
./scripts/test.sh

# Run with coverage
pytest --cov=src/atria_cli
```

### Code Formatting

```bash
# Format code
./scripts/format.sh

# Lint code
./scripts/lint.sh
```

## 🔧 Configuration

The library uses environment-based configuration. Key settings include:

- `ATRIAX_URL`: Base URL for the Atriax Hub API
- `ATRIAX_ANON_KEY`: Anonymous API key for public access

Configuration is managed through the `atria_cli.config.settings` module.

## 📁 Project Structure

```
atria_cli/
├── src/atria_cli/           # Main package
│   ├── api/                 # API modules
│   │   ├── auth.py         # Authentication API
│   │   ├── datasets.py     # Dataset management
│   │   ├── models.py       # Model management
│   │   └── ...
│   ├── client.py           # HTTP client wrapper
│   ├── config.py           # Configuration management
│   ├── hub.py              # Main AtriaHub class
│   └── models.py           # Data models
├── scripts/                # Utility scripts
└── tests/                  # Test suite
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure code quality (`./scripts/test.sh && ./scripts/lint.sh`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## 👥 Authors

- **Saifullah** - *Initial work* - [saifullah.saifullah@dfki.de](mailto:saifullah.saifullah@dfki.de)

## 🔗 Links

- [Homepage](https://github.com/saifullah3396/atria_cli/)
- [Bug Reports](https://github.com/saifullah3396/atria_cli/issues)
- [Source Code](https://github.com/saifullah3396/atria_cli/)

## 📋 Dependencies

### Core Dependencies
- `atriax-client`: Core API client
- `supabase`: Authentication and database client
- `lakefs`: Data lake file system
- `pydantic-settings`: Configuration management
- `keyring`: Secure credential storage
- `tqdm`: Progress bars for file operations

### Development Dependencies
- `pytest`: Testing framework
- `coverage`: Test coverage
- `ruff`: Linting and formatting
- `mypy`: Type checking

---

For more detailed documentation and examples, visit the [project repository](https://github.com/saifullah3396/atria_cli/).