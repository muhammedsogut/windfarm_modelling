import numpy as np
import pandas as pd

def read(numturb, path, distance=None, angle=None, wind_speed=None, location_path=None):
    dataset_pos=np.empty(shape=(0, 2))
    dataset_ws=np.empty(shape=(0, 1))
    dataset_id=np.empty(shape=(0, 1))
    
    if numturb==2:
        for dist in distance:
            for ang in angle:
                df=pd.read_csv('%s/turbine_%sp00_%sp00_report_%sD.csv' %(path, ang, wind_speed, dist), sep=' ', header=None)
                df=df.tail(1)
                df=pd.melt(df)
                df=df.dropna()
                a=df.to_numpy(dtype=float)
                a=a[:,1]
                ws=np.array([[a[22]],[a[19]]])
                position=np.array([[0., 0.], [((93.0*dist)*np.cos(np.deg2rad(270-ang))),((93.0*dist)*np.sin(np.deg2rad(270-ang)))]])
                turb_id=str('%s_turbine_%s_degrees_%s_meters' %(numturb, ang, 93.0*dist))
                print(turb_id)
                dataset_pos=np.append(dataset_pos,position,axis=0)
                dataset_ws=np.append(dataset_ws,ws,axis=0)
                dataset_id=np.append(dataset_id,turb_id, axis=0)
        dataset=np.concatenate((dataset_pos, dataset_ws, dataset_id),axis=1)
        dataset=pd.DataFrame(dataset, columns = ['X_coord','Y_coord','Ref_Wind_Speed', 'ID'])
        dataset['Num_tot_turb']=numturb
        
        
    else:
        df_loc = pd.read_csv('%s' %location_path, sep=',', index_col=0, header=0)
        for i in range(len(df_loc)):
            df=pd.read_csv('%s/%s.csv' %(path,i), sep=' ', header=None)
            df=df.tail(1)
            df=pd.melt(df)
            df=df.dropna()
            a=df.to_numpy(dtype=float)
            a=a[:,1]
            ws=np.array([[a[23]],[a[29]],[a[26]]])
            position = np.array([[df_loc.at[i,'X0'],df_loc.at[i,'Y0']],[df_loc.at[i,'X1'],df_loc.at[i,'Y1']],[df_loc.at[i,'X2'],df_loc.at[i,'Y2']]])
            dataset_pos=np.append(dataset_pos,position,axis=0)
            dataset_ws=np.append(dataset_ws,ws,axis=0)
        dataset=np.concatenate((dataset_pos, dataset_ws),axis=1)
        dataset=pd.DataFrame(dataset, columns = ['X_coord','Y_coord','Ref_Wind_Speed'])
        dataset['Num_tot_turb']=numturb
        dataset
        
    return dataset