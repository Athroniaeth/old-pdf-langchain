from pathlib import Path

# Add globals paths of the project
PROJECT_PATH = Path(__file__).absolute().parents[1]
SOURCE_PATH = PROJECT_PATH / "src"
ENV_PATH = PROJECT_PATH / '.env'

# Add paths for the project
DATA_PATH = PROJECT_PATH / "data"
DB_VECTOR_PATH = DATA_PATH / "vector"
