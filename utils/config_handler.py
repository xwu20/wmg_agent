import random
import time
import os, sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_OVERRIDES_PATH = os.path.join(os.path.dirname(CODE_DIR), 'config.py')

config_rand = random.Random(time.time())  # For 'truly' random hyperparameter selection.

'''
Configuration settings in wmg_agent:
Designed to minimize the passing of arguments through the commandline and down function call trees.
This file creates the global ConfigHandler object (cf), which other files read.
The config.py file defines every parameter and its allowed settings.
    This .py extension is only for pretty formatting. The config.py file is never actually run.
    Each line must follow one of these formats:
        1. Blank
        2. Comment (starting with #)
        3. "parameter name = setting value(s)  # Optional comment"
Parameters which are accessed during initialization of a run are written to the top of the output file.
'''

class Setting(object):
    def __init__(self, name_str, value_str, line_num):
        self.value_was_read = False
        self.line_num = line_num
        self.name_str = name_str
        self.default_value = self.set_value(value_str)

    def set_value(self, value_str):
        values = value_str.split(',')
        if len(values) > 1:
            value_str = values[config_rand.randint(0, len(values)-1)].strip()
        if value_str == 'None':
            new_value = None
        elif value_str == 'True':
            new_value = True
        elif value_str == 'False':
            new_value = False
        elif value_str == 'randint':
            new_value = config_rand.randint(0, 999999999)
        else:
            try:
                new_value = int(value_str)
            except ValueError:
                try:
                    new_value = float(value_str)
                except ValueError:
                    new_value = value_str
        self.value = new_value
        return new_value

    def get_value(self):
        self.value_was_read = True
        return self.value

class ConfigHandler(object):
    def __init__(self):
        # Get the default config values.
        self.use_xtlib = False
        self.read_config_file(open(CONFIG_OVERRIDES_PATH, 'r'))

    def read_config_file(self, file):
        self.settings = {}
        self.lines_or_settings_to_output = []
        line_num = 0
        for line in file:
            line = line[:-1]
            line_num += 1
            if (len(line) == 0) or (line[0] == '#'):
                self.lines_or_settings_to_output.append(line)
                continue
            halves = line.split('=')
            name_string, value_string = halves[0].strip(), halves[1].strip()
            if '#' in value_string:
                value_string = value_string[:value_string.index('#')].strip()
            assert name_string not in self.settings  # No duplicates allowed.
            setting = Setting(name_string, value_string, line_num)
            self.settings[name_string] = setting
            self.lines_or_settings_to_output.append(setting)

    def val(self, name):
        return self.settings[name].get_value()

    def output_to_file(self, file_name):
        section_header = None
        file = open(file_name, 'a')
        for l_or_s in self.lines_or_settings_to_output:
            if isinstance(l_or_s, str):
                section_header = l_or_s
                if (l_or_s is not None) and (l_or_s != '') and (l_or_s[1] == '#'):
                    file.write('\n' + section_header + '\n')
            else:
                if l_or_s.value_was_read:
                    if section_header is not None:
                        file.write('\n' + section_header + '\n')
                        section_header = None
                    sz = "{} = {}".format(l_or_s.name_str, l_or_s.value)
                    if l_or_s.value != l_or_s.default_value:
                        sz += "  #################### {} {} => {}".format(l_or_s.name_str, l_or_s.default_value, l_or_s.value)
                    file.write(sz + '\n')
        file.write('\n')
        file.close()

    def output_to_xt(self):
        for line in self.lines_or_settings_to_output:
            if not isinstance(line, str):
                if line.value_was_read:
                    self.xt_run_logger.log_hparams({line.name_str: line.value})

    def safe_environ_var(self, key):
        return os.environ[key] if key in os.environ else None

    def consult_azure_for_config(self):
        self.use_xtlib = True
        from xtlib.run import Run as XTRun
        self.xt_run_logger = XTRun()
        self.xt_run_logger.store.download_file_from_experiment(self.xt_run_logger.ws_name, self.xt_run_logger.exper_name, "config.txt", "config.txt")
        self.read_config_file(open("config.txt", 'r'))
        USE_DGD = self.val("USE_DGD")
        USE_DGD2 = self.val("USE_DGD2")
        if USE_DGD:
            # Use Distributed Grid Descent to configure this run.
            from xt.dgd import DGD
            dgd = DGD()
            chosen_runset = dgd.choose_config()
            dgd.apply_config(chosen_runset, self)
        if USE_DGD2:
            # Use Distributed Grid Descent to configure this run.
            from xt.dgd2 import DGD2
            dgd = DGD2()
            chosen_runset = dgd.choose_config()
            dgd.apply_config(chosen_runset, self)


cf = ConfigHandler()
