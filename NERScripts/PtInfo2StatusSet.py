import csv


def main():
    data = get_gold_data()
    statuses=set()
    for d in data:
        if len(d) > 7:
            status = d[7]
            if "DxResult" not in status:
                statuses.add(status)

    for status in sorted(statuses):
        print status


def get_gold_data():
    gold = list()
    with open('../NERResources/RawData/300PtsWithDxInfo.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            gold.append(row)
    return gold

if "__main__" == __name__:
    main()