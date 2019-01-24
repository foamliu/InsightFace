angles_file = 'data/angles.txt'
print('Calculating threshold...')

with open(angles_file, 'r') as file:
    lines = file.readlines()

data = []

for line in lines:
    tokens = line.split()
    angle = float(tokens[0])
    type = int(tokens[1])
    data.append({'angle': angle, 'type': type})

min_error = 6000
min_threshold = 0

for d in data:
    threshold = d['angle']
    type1 = len([s for s in data if s['angle'] <= threshold and s['type'] == 0])
    type2 = len([s for s in data if s['angle'] > threshold and s['type'] == 1])
    num_errors = type1 + type2
    if num_errors < min_error:
        min_error = num_errors
        min_threshold = threshold

print(min_error, min_threshold)
