# Neal Haonan Chen (hc4pa@virginia.edu)
# University of Virginia
# GDPR Prject
# 10/8/2018

# This script converts raw text in OPP115 dataset to usable machine learning dataset.
import os
import csv
import re

# Parameters
EXP = '"(.*?)": {"endIndexInSegment": ([-0-9]*), "startIndexInSegment": ([-0-9]*)(, "selectedText": ")?(.*?)"?, "value": "(.*?)"}'
PATH = "" # Files path
OUTPUT_DATA = "sample.txt" # Name of data output file
OUTPUT_LABEL = "sample2.txt" # Name of label output file

# Variables
list_cast = []      # List of categories
dict = {}
dict_data = open(OUTPUT_DATA, "w")
dict_label = open(OUTPUT_LABEL, "w")
pattern = re.compile(EXP)

def process(cat):
    return cat.split()[0]

# Read every file in directory
for filename in os.listdir(PATH):
    with open("sample/"+filename, "r") as infile:
        reader = csv.reader(infile)
        for rows in reader:
            result = pattern.findall(rows[6])
            for item in result:
                if item[1] == "-1":
                    continue
                if item[-2] in dict:
                    if process(item[0]) not in dict[item[-2]]:
                        dict[item[-2]].append(process(item[0]))
                else:
                    dict[item[-2]] = [process(item[0])]

# After extracting data and storing them in the RAM, create output files print them to the files.
for key in dict.keys():
    temp = ""
    for l in dict[key]:
        if l not in list_cast:
            list_cast.append(l)
        temp += l+ " "
    dict_data.write(temp + "\n")
    dict_data.write(key.lower() + "\n")
for item in list_cast:
    print(process(item))
    dict_label.write(item + '\n')

print("Process finished: # of categories:",len(list_cast))
