import csv

with open('ratings.csv', encoding='UTF-8') as f:
    rating_csv = csv.reader(f)
    headers = next(rating_csv)
    print(headers)
    with open('test_data.csv', "w", encoding="utf-8", newline="") as w:
        data = csv.writer(w)
        data.writerow(headers)
        line = [0, 0, 0, 0, 0]
        for i in range(1, 6):
            for row in rating_csv:
                if row[2] == str(i) and line[i - 1] <= 3000:
                    if len(row[4]) > 60:
                        row[4] = row[4].replace("\n", "")
                        data.writerow(row)
                        line[i - 1] = line[i - 1] + 1
                if line[i - 1] == 3000:
                    break
