import scipy.io as sio
import pickle

file_paths = {
    "subsonic_1": "Raw Data/241111_Group07_Subsonic_Total_1.mat",
    "subsonic_2": "Raw Data/241111_Group07_Subsonic_Total_2.mat",
    "subsonic_3": "Raw Data/241111_Group07_Subsonic_Total_3.mat",
    "subsonic_4": "Raw Data/241111_Group07_Subsonic_Total_4.mat",
    "subsonic_5": "Raw Data/241111_Group07_Subsonic_Total_5.mat",
    "subsonic_6": "Raw Data/241111_Group07_Subsonic_Total_6.mat",
    "subsonic_7": "Raw Data/241111_Group07_Subsonic_Total_7.mat",
    "supersonic_1": "Raw Data/241111_Group07_Supersonic_Total_1.mat",
    "supersonic_2": "Raw Data/241111_Group07_Supersonic_Total_2.mat",
    "supersonic_3": "Raw Data/241111_Group07_Supersonic_Total_3.mat",
    "supersonic_4": "Raw Data/241111_Group07_Supersonic_Total_4.mat",
    "supersonic_5": "Raw Data/241111_Group07_Supersonic_Total_5.mat",
    "supersonic_6": "Raw Data/241111_Group07_Supersonic_Total_6.mat",
    "supersonic_7": "Raw Data/241111_Group07_Supersonic_Total_7.mat",
}

data_dict = {}


for label, path in file_paths.items():
    data = sio.loadmat(path)
    
    ActualScanRate = int(data['ActualScanRate'][0][0])
    currentlocation = int(data['currentlocation'][0][0])
    dataP = data['dataP']
    dataV = data['dataV']
    v_offset_mean = data['v_offset_mean'][0]
    
    data_dict[label] = {
        'ActualScanRate': ActualScanRate,
        'currentlocation': currentlocation,
        'dataP': dataP,
        'dataV': dataV,
        'v_offset_mean': v_offset_mean
    }


with open('data_dict.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

print("Data has been saved to 'data_dict.pkl'")