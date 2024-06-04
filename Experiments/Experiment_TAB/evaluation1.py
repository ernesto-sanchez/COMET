#Script to get the evaluation of the peorfomance of the models when the TAB paramenter varies from 0 to 1.



import configparser
import os


project_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Create a config parser
config = configparser.ConfigParser()

# Read the config file
config_path = os.path.join(os.path.dirname(__file__), 'config1.ini') 
config.read(config_path)

for tab in [0, 0.2, 0.4, 0.6, 0.8, 1]:
    config['synthetic_data']['TAB'] = str(tab)
    with open(config_path, 'w') as configfile:
        config.write(configfile)
    file = os.path.join(project_path, 'evaluation', 'evaluation.py')
    os.system(f"python {file}")

