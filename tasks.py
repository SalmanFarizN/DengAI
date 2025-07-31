import csv
from datetime import datetime
import os


def concat_csv(file1_path, file2_path, output_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"combined_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)

    with open(file1_path, newline='', encoding='utf-8') as f1, \
         open(file2_path, newline='', encoding='utf-8') as f2, \
         open(output_path, 'w', newline='', encoding='utf-8') as fout:

        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        writer = csv.writer(fout)

        header = next(reader1)
        writer.writerow(header)

        for row in reader1:
            writer.writerow(row)

        next(reader2)  # skip second file's header
        for row in reader2:
            writer.writerow(row)

    print(f"Files concatenated and saved to: {output_path}")

# concat_csv('src/predictions1.csv', 'src/predictions.csv', "./")