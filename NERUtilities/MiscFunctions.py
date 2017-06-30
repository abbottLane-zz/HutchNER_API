import ConfigParser
import os

CLUSTER_PATH =os.path.join(os.path.dirname(__file__), os.path.join("..", "NERResources","Cluster_Files","MMC867k_FH255k.600.cbow.model.bin_k=800minibatch=False.kmeans"))

def load_labkey_server_info_from_ini(ini_section_entry):
    """
    Given a section of the labkey.ini file, loads all configurations associated with that entry.
    :param ini_section_entry: string name of the .ini section
    :return: a dictionary of relevant configuration field names and values
    """
    config = ConfigParser.ConfigParser()
    # HutchNER1/HutchNER1/NERResources/labkey.ini
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    labkey_path = ROOT_DIR + os.sep + ".."+os.sep+"NERResources" + os.sep+ "labkey.ini"
    if not os.path.isfile(labkey_path):
        raise ValueError("Error: Labkey path is not a real file location:", labkey_path)
    else:
        try:
            config.read(labkey_path)
        except:
            raise ValueError(
                "Error reading from config file at: ./NERResources/labkey.ini \n\tMake sure this file exists!")
    err_message = "Retrieved: ["
    try:
        driver = config.get(ini_section_entry, "Driver")
        err_message += "driver, "
        database = config.get(ini_section_entry, "Database")
        err_message += "database, "
        server = config.get(ini_section_entry, "Server")
        err_message += "server, "
        table = config.get(ini_section_entry, "Table")
        err_message += "table, "
        job_run_ids = config.get(ini_section_entry, "JobRunIds").split(",")
        err_message += "jobRunIds."
    except:
        err_message += "] \n"
        raise ValueError(
            err_message + "Error parsing the \'NERResources/labkey.ini\' file: Make sure the section entry [" + ini_section_entry + "] " +
                "\n\texists, and contains these 5 entries: \'Driver\', \'Database\', \'Server\', \'Table\',"
                "\'JobRunIds\'")
    return {"driver": driver, "database": database, "server": server, "table": table, "job_run_ids": job_run_ids}
