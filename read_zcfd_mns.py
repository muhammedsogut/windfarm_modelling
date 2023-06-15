import numpy as np
import pandas as pd

def read(numturb, path, diameter_distance=None, distance=None, angle=None, wind_speed=None, location_path=None):
    dataset=pd.DataFrame()
    
    if numturb==2:
        if distance==None:
            for dist in diameter_distance:
                for ang in angle:
                    df=pd.read_csv('%s/turbine_%sp00_%sp00_report_%sD.csv' %(path, ang, wind_speed, dist), sep=' ', header=None)
                    df=df.tail(1)
                    turb_id=str('%s_turbine_%s_degrees_%s_meters' %(numturb, ang, 93.0*dist))
                    turb_data=pd.DataFrame([[0., 0., float(df.iloc[0,62]), turb_id, 0, wind_speed],[((93.0*dist)*np.cos(np.deg2rad(270-ang))),((93.0*dist)*np.sin(np.deg2rad(270-ang))), float(df.iloc[0,53]), turb_id, 1, wind_speed]],columns=['x_coord','y_coord','ref_wind_speed','id', 'turb_num', 'wind_speed'])
                    dataset=dataset.append(turb_data)
            dataset['Num_tot_turb']=numturb
            
        elif diameter_distance==None:
            for dist in distance:
                for ang in angle:
                    df=pd.read_csv('%s/turbine_%sp00_%sp00_report_%s.csv' %(path, ang, wind_speed, dist), sep=' ', header=None)
                    df=df.tail(1)
                    turb_id=str('%s_turbine_%s_degrees_%s_meters' %(numturb, ang, dist))
                    turb_data=pd.DataFrame([[0., 0., float(df.iloc[0,62]), turb_id, 0, wind_speed],[((dist)*np.cos(np.deg2rad(270-ang))),((dist)*np.sin(np.deg2rad(270-ang))), float(df.iloc[0,53]), turb_id, 1, wind_speed]],columns=['x_coord','y_coord','ref_wind_speed','id', 'turb_num', 'wind_speed'])
                    dataset=dataset.append(turb_data)
            dataset['Num_tot_turb']=numturb
        
        
    else:
        df_loc = pd.read_csv('%s' %location_path, sep=',', index_col=0, header=0)
        for i in range(len(df_loc)):
            df=pd.read_csv('%s/%s.csv' %(path,i), sep=' ', header=None)
            df=df.tail(1)
            turb_id= str('%s_turbine_%sth_simulation' %(numturb, i))
            turb_data= pd.DataFrame([[df_loc.at[i,'X0'],df_loc.at[i,'Y0'], float(df.iloc[0,65]), turb_id, 0, wind_speed],[df_loc.at[i,'X1'],df_loc.at[i,'Y1'], float(df.iloc[0,83]), turb_id, 1, wind_speed],[df_loc.at[i,'X2'],df_loc.at[i,'Y2'], float(df.iloc[0,74]), turb_id, 2, wind_speed]], columns=['x_coord','y_coord','ref_wind_speed','id', 'turb_num', 'wind_speed'])
            dataset=dataset.append(turb_data)
        dataset['Num_tot_turb']=numturb
        
    return dataset