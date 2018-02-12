import numpy as np
import pandas as pd
import os
import scipy.io as sio

def extract(fname, give=False):
    try:
        dat = sio.loadmat(fname)
    except:
        return 'Bad file'
    raw_data = dat['data']

    # get list of things in the .mat file and check if they are a data array
    things = raw_data.dtype.fields.keys()
    lens = [raw_data[thing][0][0].shape[0] for thing in things]

    # get the length of the data arrays
    length = max(lens)
    if(length < 10):
        return 'Nothing of substance'

    # look at just the data arrays
    varnames = [thing for thing in things if raw_data[thing][0][0].shape[0]==length]

    # find energy
    engname = [thing for thing in things if thing[:4]=='BEND'][0]
    eng = int(raw_data[engname][0][0])

    # build matrix, putting gdet last
    mat = np.zeros(shape=(length,len(varnames)))
    n_added = 0
    cols = []
    last_col = ''
    for name in varnames:
        if(name[:4]=='GDET'):
            mat[:,[-1]] = raw_data[name][0][0]
            last_col = name
        else:
            try:
                mat[:,[n_added]] = raw_data[name][0][0]
            except:
                return 'Weird error.'
            n_added += 1
            cols.append(name)
    if(last_col==''):
        return 'No GDET'
    cols.append(last_col)

    # build and write dataframe without NaNs
    df = pd.DataFrame(data=mat, columns=cols)
    df = df.loc[~np.isnan(df).any(axis=1),:]
    if(not give):
        df.to_csv(fname[:-4] + '_' + str(eng) + '.csv',index=False)
        return 'Success'
    else:
        return df

def dir_to_csv(dir_name):
    files = [f for f in os.listdir(dir_name) if f[-3:]=='mat']
    for f in files:
        extract(dir_name+f)
