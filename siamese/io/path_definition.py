import os
from typing import Dict

def get_project_dir() -> str:

    dir_as_list = os.path.dirname(__file__).split("/")
    index = dir_as_list.index("apparel-datascience-siamese")
    project_directory = f"/{os.path.join(*dir_as_list[:index+1])}"

    return project_directory


def get_tf_record_default_parameters() -> Dict:

    dir_train = f"{get_project_dir()}/data/train"

    train_csv_path = f"{dir_train}/image_train_list.csv"
    val_csv_path = f"{dir_train}/image_val_list.csv"
    test_csv_path = f"{dir_train}/image_test_list.csv"
    image_directory = '/home/storage/cdn/images/source'

    num_shards = 128

    default_path = {"TRAIN_CSV_PATH": train_csv_path,
                    "VAL_CSV_PATH": val_csv_path,
                    "TEST_CSV_PATH": test_csv_path,
                    "NUM_SHARDS": num_shards,
                    "IMAGE_DIRECTORY": image_directory}

    return default_path


def get_file(relative_path: str) -> str:

    """
    Get the full path

    Args:
        relative_path:

    Returns:
    """

    return f"{get_project_dir()}/{relative_path}"

