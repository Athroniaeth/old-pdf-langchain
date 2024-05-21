import sys
from pathlib import Path

from dotenv import load_dotenv

project_path = Path(__file__).parents[1].absolute()
sys.path.append(f"{project_path}")

from src import ENV_PATH  # noqa: E402
from src.app import launch_gradio  # noqa: E402


def main():
    launch_gradio()


if __name__ == "__main__":
    # Load the environment variables
    load_dotenv(ENV_PATH)

    # Run the main function
    main()
