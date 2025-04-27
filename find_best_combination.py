def parse_line(line):
    parts = line.strip().split('|')
    hyperparams = parts[0].strip().split(',')
    metrics = parts[1:]
    
    alpha = float(hyperparams[0].split('=')[1])
    onset = float(hyperparams[1].split('=')[1])
    offset = float(hyperparams[2].split('=')[1])
    der = float(metrics[3].split(':')[1].strip())

    return {'alpha': alpha, 'onset': onset, 'offset': offset, 'DER': der, 'line': line.strip()}

def find_best(lines, alpha_condition):
    best = None
    for line in lines:
        if not line.strip():
            continue
        parsed = parse_line(line)
        if alpha_condition(parsed['alpha']):
            if best is None or parsed['DER'] < best['DER']:
                best = parsed
    return best

print("For 2s:")
with open('DER_results_2s.txt', 'r') as f:
    lines = f.readlines()

# Set condition for alpha == 1
best_eq_1 = find_best(lines, lambda a: a == 1)

# Set condition for alpha != 1
best_neq_1 = find_best(lines, lambda a: a != 1)

if best_eq_1:
    print("Best for alpha == 1:\n", best_eq_1['line'])
else:
    print("No entries with alpha == 1")

if best_neq_1:
    print("Best for alpha != 1:\n", best_neq_1['line'])
else:
    print("No entries with alpha != 1")

print("\n\nFor as")

with open('DER_results_all_speaker.txt', 'r') as f:
    lines = f.readlines()

# Set condition for alpha == 1
best_eq_1 = find_best(lines, lambda a: a == 1)

# Set condition for alpha != 1
best_neq_1 = find_best(lines, lambda a: a != 1)

if best_eq_1:
    print("Best for alpha == 1:\n", best_eq_1['line'])
else:
    print("No entries with alpha == 1")

if best_neq_1:
    print("Best for alpha != 1:\n", best_neq_1['line'])
else:
    print("No entries with alpha != 1")

