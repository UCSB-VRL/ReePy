# ReebPy

ReebPy is a Python library for building Reeb graphs from sequence data. This project is managed using [Poetry](https://python-poetry.org/), which simplifies dependency management and packaging.

![Python Version](https://img.shields.io/badge/python-3.11-blue)

## Installation

To get started with ReebPy, follow these steps to set up your development environment.

### Prerequisites

Ensure you have Python 3.11 installed on your system. You can verify your Python version by running:

```bash
python --version
```

### Installing Poetry

If you haven't already installed Poetry, you can do so by running:

```bash
curl -sSL https://install.python-poetry.org | python3
```

### Setting Up the Project

1. **Clone the Repository**:

   Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/reebpy.git
   cd reebpy
   ```

2. **Install Dependencies**:

   Use Poetry to install the project dependencies:

   ```bash
   poetry install
   ```

   This command will create a virtual environment and install all the necessary dependencies specified in the `pyproject.toml` file.

3. **Activate the Virtual Environment**:

   To activate the virtual environment, run:

   ```bash
   poetry shell
   ```

   This will allow you to run commands within the context of the virtual environment.
