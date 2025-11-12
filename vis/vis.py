import numpy as np
import os
import matplotlib.pyplot as plt
plt.rc('font', family = 'sans-serif')
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
if not os.path.exists('figure'):
    os.makedirs('figure')

# read the query_list_all.json
with open('query_list_all.json', 'r') as f:
    query_list_all = f.read()
    # get a list of the _id oid and the description from the json
    import json
    query_list_all = json.loads(query_list_all)
    query_list_all = [{"oid": item["_id"]["$oid"], "description": item["description"]} for item in query_list_all]
    print(query_list_all)
# get all the files in data folder with .npy extension
data_files = [f for f in os.listdir('data') if f.endswith('.npy')]
figure_files = [f for f in os.listdir('figure') if f.endswith('.pdf')]
# delete the idata_files that are already in figure_files
for f in figure_files:
    data_file = f[:-4] + '.npy'
    if data_file in data_files:
        data_files.remove(data_file)
# sort the files
data_files.sort()
for file in data_files:
    sf = np.load(f'data/{file}')
    plt.figure(figsize=(40, 8))
    plt.rcParams['font.size'] = 20
    plt.imshow(3600*sf[:, :]/5280, aspect='auto', cmap='Greys_r',vmin=0, vmax=80)
    plt.colorbar(label='Speed (mph)', pad=0.01, fraction=0.02)
    # get the oid from the file name
    oid = file[:-14]
    lane = int(file[-9:-8])
    # get the description from the query_list_all
    description = ''
    for item in query_list_all:
        if item['oid'] == oid:
            description = item['description']
            break
    plt.title(f'Lane {lane} | {description}', fontsize=25)
    # plt.title(f'{file[:-4]}', fontsize=30)
    plt.savefig(f'figure/{file[:-4]}.pdf', bbox_inches='tight',dpi=300, format='pdf')
    plt.close()