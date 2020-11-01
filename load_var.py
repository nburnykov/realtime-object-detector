import os
import json
from os.path import join


class C:
    def __init__(self, config_path):
        with open(join(config_path), 'r') as f:
            self.config = json.load(f)

    def load_variable(self, var_name: str, default_value=None, var_type=str):
        variable = os.getenv(var_name)
        if variable is not None:
            if var_type == 'json':
                variable = json.loads(variable)
            elif var_type is bool:  # caution: different logic for handling bool vars
                variable = variable.lower() in ['true', '1', 'yes']
            else:
                variable = var_type(variable)
        else:
            variable = self.config.get(var_name)

        if variable is None:
            variable = default_value

        return variable
