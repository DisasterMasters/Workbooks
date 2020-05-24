import os
import sys
import pymongo
import datetime
from pprint import pprint
from tqdm import tqdm_notebook as tqdm
import text_processing_helper as h

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: python3 {} sample_size".format(sys.argv[0]))
    sys.exit()
    
try: 
    sample_size = int(sys.argv[1])
    os.mkdir('keyword_search_samples/{}'.format(sample_size))
except:
    print("Usage: python3 {} sample_size".format(sys.argv[0]))
    

# connect to MongDB
db = pymongo.MongoClient("da1.eecs.utk.edu")["twitter"]
coll_name = "statuses_a:covid19"
coll = db[coll_name]
print("\nCollection: {}\nNumber_Count: {}".format(coll_name, coll.estimated_document_count()))

keywords = [
    "low income",
    "income",
    "utility",
    "utility companies",
    "utility company",
    "utilities bill burden",
    "utilities cost burden",
    "utility bill burden",
    "utility cost burden",
    "cost burden",
    "bill burden",
    "utilities",
    "electricity",
    "water",
    "internet",
    "home environment",
    "home energy environment",
    "no heater",
    "no heat",
    "no air conditioning",
    "no ac",
    "inequality",
    "social distancing",
    "utilities disconnected notices",
    "cannot afford utilities bills",
    "can not pay utilities",
    "Low Income Home Energy Assistance Program",
    "LIHEAP",
    "CARES act",
    "Utility bill debt crisis",
    "Unemployment and utilities bills",
    "Shut-off moratorium",
    "Utility customer debt relief",
    "disconnect electricity",
    "disconnect notice",
    "Spam and utilities",
    "Negative credit reporting",
    "Social housing",
    "low-income housing",
    "low-income buildings",
    "unsafe homes",
    "unrepaired homes",
    "vulnerable population",
    "energy burden",
    "energy insecurity",
    "energy poverty",
    "energy bills",
    "Residential energy consumption increase",
    "Weatherization Assistance Program",
    "WAP",
    "Renters",
    "low-income households",
    "low-income families",
    "impacts on low-income",
    "climate change",
    "climatechange",
    "global warming",
    "globalwarming",
    "carbon dioxide",
    "emission",
    "ecosystem",
    "sustainability",
    "electric power",
    "electric outage",
    "electricity",
    "electricity out",
    "power loss",
    "power recovery",
    "power outage",
    "power shortage",
    "power problem",
    "no light",
    "no power",
    "no electricity",
    "carbon emissions",
    "electricity price",
    "electricity cost",
    "utility cost"
    
]

keywords = list(set(keywords))

texts = []
print("getting {} samples".format(sample_size))
#for i in coll.aggregate([{"$sample": {"size": 10000}}]):
for i in coll.find({"lang" : "en"}, limit=sample_size):
    #pprint(i['text'])
    #print()
    texts.append(i['text'])
print("done\n")

print("searching keywords")
d = {}
for k in keywords:
    h.phase_appearance(texts, k, d=d)
print("done\n")

print("writing to keyword_search_{}_sample.txt".format(sample_size))
f = open("keyword_search_samples/{}/keyword_search_{}_sample.txt".format(sample_size, sample_size), "w")
f.write("Collection name: {}\n".format(coll_name))
f.write("Collection size: {}\n".format(coll.estimated_document_count()))
f.write("Sample size: {}\n\n".format(sample_size))
for k in d.keys():
    num = len(d[k])
    f.write("{}: {}, {}\n".format(k, num, num/sample_size))
f.close()


# plotting
height = []
keywords_found = []

for k in d.keys():
    if len(d[k])!=0:
        keywords_found.append(k)
        height.append(len(d[k]))

# save xlsx file 
with pd.ExcelWriter("keyword_search_samples/{}/keyword_search_{}_sample.xlsx".format(sample_size, sample_size)) as writer:
    for k in keywords_found:
        text_temp = []
        temp_df = pd.DataFrame()
        for i in d[k]:
            text_temp.append(texts[i])
        temp_df['Text'] = text_temp
        temp_df.to_excel(writer, sheet_name='{}'.format(k))
        
# Graphing
bars = keywords_found
y_pos = np.arange(len(bars))

# Create horizontal bars
plt.barh(y_pos, height, alpha=0.5)

# Create names on the y-axis
plt.yticks(y_pos, bars, fontsize=8)

for i, v in enumerate(y_pos):
    plt.text(height[v]+0.1, i-0.1, "{}%".format(height[v]*100/sample_size), color='blue', fontsize=8)

# Save hist graphic
plt.tight_layout()
plt.title("Keywords Searching (sample size:{})".format(sample_size))
plt.savefig("keyword_search_samples/{}/keyword_search_{}_sample.png".format(sample_size, sample_size))
plt.clf()

# second graph
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = bars[:]
sizes = []
explode = []

y_pos = np.arange(len(labels))
for i, v in enumerate(y_pos):
    labels[v] = labels[v] + ": " + str(height[v]*100/sample_size) + "%"
    sizes.append(height[v])
    explode.append(0.05)

#fig1, ax1 = plt.subplots()
plt.title("Keywords Searching (sample size:{})\n".format(sample_size))
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.savefig("keyword_search_samples/{}/keyword_search_{}_sample_pie.png".format(sample_size, sample_size))
#plt.show()

print("done")