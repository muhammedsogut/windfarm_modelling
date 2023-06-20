import numpy as np
import pandas as pd

def read(numturb, filename, windspeed, position, simname):
    dataset=pd.DataFrame()
    df=pd.read_csv(filename, sep=' ')
    df=df.tail(1) # Last line is the final result
    if numturb == 2:
        turb_data=pd.DataFrame(
            [[position[0,0],position[0,1], float(df.iloc[0,62]),windspeed,0,numturb, simname],
            [position[1,0],position[1,1], float(df.iloc[0,53]),windspeed,1,numturb, simname]],
            columns=["x_coord", "y_coord", "ref_wind_speed","wind_speed" , "turb_num","num_tot_turb", "ID"])
        dataset = pd.concat([dataset, turb_data])
    else:
        for i in range(0,numturb):
            uref='T00%d_uref' %(i+1)
            turb_data=pd.DataFrame(
		        [[position[i,0],position[i,1],df[uref].iloc[-1],windspeed,i,numturb, simname]], 
		        columns=["x_coord", "y_coord", "ref_wind_speed","wind_speed" , "turb_num","num_tot_turb", "ID"])
            dataset = pd.concat([dataset, turb_data])
        
    return dataset