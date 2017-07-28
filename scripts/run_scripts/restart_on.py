import json

with open('input.json', 'r') as data_file:
    data = json.load(data_file)

data["restart"] = 1

with open('input.json', 'w') as data_file:
    data_file.write(json.dumps(data, indent=1) + "\n")
