import numpy as np
import csv

new_rows = []

with open('Data/new_data_columns.csv', 'r') as csvfile:
  reader = csv.reader(csvfile, delimiter=',', quotechar='"')

  ref = {}
  for row in reader:
    try:
      ref[row[0][:-4]] = float(row[4])
    except:
      print('err')

  print(ref)

# with open('Data/new_data_columns.csv', 'r') as csvfile:
#   reader = csv.reader(csvfile, delimiter=',', quotechar='"')

#   ref = {}
#   for row in reader:
#     try:
#       ref[row[0]] = float(row[1])
#     except:
#       print('err')

#   print(ref)

# with open('Data/new_pop_data.csv', 'w') as csvfile:
#   writer = csv.writer(csvfile, delimiter=',', quotechar='"', lineterminator='\n')

#   for row in new_rows:
#     writer.writerow(row)
