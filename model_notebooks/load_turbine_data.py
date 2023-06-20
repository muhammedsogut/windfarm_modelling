import pandas as pd
import os

def load_turbine_data(numturb, wind_speed, path, coord_data_path):
    df_loc = pd.read_csv(coord_data_path, sep=',', index_col=0, header=0)

    turb_data = pd.DataFrame()  # Create an empty DataFrame to store the turbine data

    for filename in os.listdir(path):
        if filename.endswith('.csv'):  # Only consider CSV files
            basename = int(os.path.splitext(filename)[0])  # Remove file extension
            
            # Check if the basename exists in the coordinate data DataFrame
            if basename in df_loc.index:
                filepath = os.path.join(path, filename)  # Construct the full file path
                
                # Call the read_zcfd function to read turbine data
                data = read_zcfd.read(numturb, filepath, wind_speed,
                                      np.array([[df_loc.at[basename,'X0'],df_loc.at[basename,'Y0']],
                                                [df_loc.at[basename,'X1'],df_loc.at[basename,'Y1']],
                                                [df_loc.at[basename,'X2'],df_loc.at[basename,'Y2']]]
                                              ),
                                      str('%s_turbine_%sth_simulation' %(numturb, basename)))
                
                turb_data = pd.concat([turb_data, data])  # Concatenate the data to the turb_data DataFrame
            else:
                print(f"Warning: Coordinate data not found for {basename}. Skipping file.")
    
    return turb_data