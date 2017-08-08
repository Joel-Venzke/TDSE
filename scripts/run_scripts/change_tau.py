import json
import sys

# read current input file
with open('input.json', 'r') as data_file:
    data = json.load(data_file)

# turn on restart
data["laser"]["pulses"][1]["tau_delay"] = sys.argv[1];

# write file
with open('input.json', 'w') as data_file:
    data_file.write(json.dumps(data, indent=1) + "\n")
