import os
import platform
import logging
import yaml
import shutil

CONFIG_FILE = os.getenv("VCAT_CONFIG", "")

def find_difmap_path(logger):
    difmap_path = shutil.which("difmap")
    if difmap_path:
        difmap_path = "/".join(difmap_path.split("/")[:-1])+"/"
        logger.info(f"Using DIFMAP Path: {difmap_path}")
    else:
        difmap_path = ""
        logger.info(f"DIFMAP not found in path, will not be able to use DIFMAP functionality.")
    return difmap_path

#load config file
def load_config(path=""):
    global difmap_path
    if path=="":

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )

        #no input file specified
        logger = logging.getLogger("vcat")
        logger.info("Logging initialized. Log file: Console only.")
        logger.info("No environment variable VCAT_CONFIG found, will use defaults.")
        difmap_path=find_difmap_path(logger)

    else:
        with open(path, "r") as file:
            config = yaml.safe_load(file)

        LOG_LEVEL = getattr(logging, config["logging"].get("level", "INFO").upper(), logging.INFO)
        LOG_FILE = config["logging"].get("log_file", None)

        if LOG_FILE:  # If log file is specified
            logging.basicConfig(
                level=LOG_LEVEL,
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                filename=LOG_FILE,
                filemode="a"  # Append mode
            )
        else:  # Log to console only
            logging.basicConfig(
                level=LOG_LEVEL,
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
            )

        logger = logging.getLogger("vcat")
        logger.info("Logging initialized. Log file: %s", LOG_FILE if LOG_FILE else "Console only")
        try:
            difmap_path = config["difmap_path"]
            logger.info(f"Using DIFMAP Path: {difmap_path}")
        except:
            difmap_path=find_difmap_path(logger)

    return logger

logger=load_config(CONFIG_FILE)