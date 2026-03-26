import csv


def sum_amounts(filename: str) -> float:
    total = 0.0
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += float(row["amount"])
    return total
