import csv
with open('300PtsWithDxInfo.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    mrns= set()
    for row in spamreader:
        items = str(row).split(',')
        mrns.add(items[1])
    with open('mrns.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for mrn in mrns:
            if mrn[0]=="U": # theres a few weird ones that start with other letters
                spamwriter.writerow([mrn[:8]]) # and one has a garbage character at the end
    print mrns