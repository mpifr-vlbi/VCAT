import os
import platform
import logging
import yaml
import shutil

CONFIG_FILE = os.getenv("VCAT_CONFIG", "")

#Initialize logging
print("\rThank you for using VCAT. Have fun with VLBI!", end="\n")
print("\rIf you are using this package please cite VCAT Team et al. 2025 ....")

def find_difmap_path(logger):
    difmap_path = shutil.which("difmap")
    if difmap_path:
        difmap_path = "/".join(difmap_path.split("/")[:-1])+"/"
        logger.info(f"Using DIFMAP path: {difmap_path}")
    else:
        difmap_path = ""
        logger.info(f"DIFMAP not found in path, will not be able to use DIFMAP functionality.")
    return difmap_path

#load config file
def load_config(path=""):
    global difmap_path
    global uvw
    global font
    global noise_method
    global mfit_err_method
    global res_lim_method
    global plot_colors
    global plot_markers
    global plot_linestyles
    global H0
    global Om0

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

        #DEFAULTS
        uvw=[0,-1]
        font="DejaVu Sans"
        noise_method="Histogram Fit"
        mfit_err_method="flat"
        res_lim_method="Kovalev05"
        plot_colors=["#023743FF","#FED789FF", "#72874EFF", "#476F84FF", "#A4BED5FF", "#453947FF"]
        plot_markers=[".",".",".",".",".","."]
        plot_linestyles=["-","-","-","-","-","-"]
        H0=67.4 #Planck Collaboration 2020
        Om0=0.315 #Planck Collaboration 2020
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
        logger.info(f"Using config file VCAT_CONFIG={path}")
        try:
            difmap_path = config["difmap_path"]
            logger.info(f"Using DIFMAP Path: {difmap_path}")
        except:
            difmap_path=find_difmap_path(logger)
        try:
            uvw = config["uvw"]
            logger.info(f"Using uv-weighting: {uvw}")
        except:
            uvw = [0,-1]

        try:
            font = config["font"]
            logger.info(f"Using font: {font}")
        except:
            font = "DejaVu Sans"

        try:
            noise_method = config["noise_method"]
            logger.info(f"Using noise method: {noise_method}")
        except:
            noise_method = "Histogram Fit"

        try:
            mfit_err_method = config["mfit_err_method"]
            logger.info(f"Using modelfit error method: {mfit_err_method}")
        except:
            mfit_err_method = "flat"

        try:
            res_lim_method = config["res_lim_method"]
            logger.info(f"Using resolution limit method: {res_lim_method}")
        except:
            res_lim_method = "Kovalev05"

        try:
            plot_colors = config["plot_colors"]
        except:
            plot_colors = ["#023743FF", "#FED789FF",  "#72874EFF", "#476F84FF", "#A4BED5FF", "#453947FF"]

        try:
            plot_markers = config["plot_markers"]
        except:
            plot_markers = [".",".",".",".",".","."]

        try:
            plot_linestyles = config["plot_linestyles"]
        except:
            plot_linestyles = ["-","-","-","-","-","-"]

        try:
            H0 = config["H0"]
        except:
            H0 = 67.4

        try:
            Om0 = config["Om0"]
        except:
            Om0 = 0.315

    return logger

global logger

logger=load_config(CONFIG_FILE)
