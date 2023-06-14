import os

images = ['clean_10_mW_9_',
 'clean_10_mW_3_',
 'clean_10_mW_5_',
 'clean_10_mW_5_',
 'clean_10_mW_7_',
 'clean_10_mW_1_',
 'clean_10_mW_3_',
 'clean_10_mW_9_',
 'clean_10_mW_1_',
 'clean_10_mW_7_']

images = [n.replace('clean','noisy') for n in images]

def get_files(path:str):
    # get only the files 
    docs = os.listdir(path)
    only_files = [f for f in docs if os.path.isfile(path+'/'+f) and '.stk' in f]
    files = []
    for file in only_files:
        for im in images:
            if im in file:
                files.append(file)

    files = list(set(files)) #remove repeated components

    return files # list of the files to open


files = get_files(snakemake.input[0])# type: ignore

with open(snakemake.output[0], 'w') as f:# type: ignore
            for line in files:
                f.write(f"{line}\n")
