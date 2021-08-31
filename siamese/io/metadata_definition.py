import configparser

from siamese.io.path_definition import get_file

def set_config():
    """
    Get all the credentials from the credential file

    Returns:
        a dictionary containing credential to different servers
    """

    config = configparser.ConfigParser()
    config.read(get_file("config/credentials.ini"))

    run_env = config['APP']['run_env']
    api_token = config['APP']['api_token']
    aws = dict(config['AWS'])
    shoe_database = dict(config['SHOE_DATABASE'])
    apparel_database = dict(config['APPAREL_DATABASE'])

    return {"RUN_ENV": run_env,
            "API_TOKEN": api_token,
            "AWS": aws,
            "MYSQL_SHOE_DATABASE_CONFIG": shoe_database,
            "MYSQL_APPAREL_DATABASE_CONFIG": apparel_database}