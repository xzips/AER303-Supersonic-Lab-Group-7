#load pickle
import pickle
with open('data_dict.pkl', 'rb') as f:
    loaded_data_dict = pickle.load(f)

# make a dataframe from a csv file
import pandas as pd
df = pd.read_csv('/content/AER303 Lab 2 Uncertainties - Sheet1.csv')
p_uncertinties = df['Unnamed: 1'].tolist()
u_subsonic = p_uncertinties[1:13]
u_supersonic = p_uncertinties[16:]
pitot_subsonic = u_subsonic.pop()
pitot_supersonic = u_supersonic.pop()
pitot_subsonic = [pitot_subsonic]*7
pitot_supersonic = [pitot_supersonic]*7
pitot_subsonic = list(map(float, pitot_subsonic))
pitot_supersonic = list(map(float, pitot_supersonic))


u_subsonic = u_subsonic[:7]
u_supersonic = u_supersonic[:7]
u_subsonic = list(map(float, u_subsonic))
u_supersonic = list(map(float, u_supersonic))


print(u_subsonic)
print(u_supersonic)


#print all key names
# print(loaded_data_dict.keys())
# print(loaded_data_dict['subsonic_1'].keys())
# print(loaded_data_dict['subsonic_2']['dataP'])


# pressure on the wall
wall_pressure = []
pitot_average = []

files = loaded_data_dict.keys()
for file_name in files:
  data = loaded_data_dict[file_name]['dataP']
  wall_loc_avg = 0
  pitot_loc_average = 0
  index = int(loaded_data_dict[file_name]['currentlocation'])
  for i in range(len(data)):
    wall_loc_avg += data[i][index - 1]
    pitot_loc_average += data[i][11]
  wall_loc_avg = wall_loc_avg/len(data)
  pitot_loc_average = pitot_loc_average/len(data)
  wall_pressure.append(wall_loc_avg)
  pitot_average.append(pitot_loc_average)

print(wall_pressure)
print(pitot_average)


sub_sonic_wall_pressure = wall_pressure[:7]
sub_sonic_pitot_average = pitot_average[:7]
super_sonic_wall_pressure = wall_pressure[7:]
super_sonic_pitot_average = pitot_average[7:]

print(len(sub_sonic_wall_pressure))
print(len(sub_sonic_pitot_average))
print(len(super_sonic_wall_pressure))
print(len(super_sonic_pitot_average))

for i in range(len(sub_sonic_wall_pressure)):
  sub_sonic_wall_pressure[i] += 101325
  sub_sonic_wall_pressure[i] /= 1000
  u_subsonic[i] /= 1000
  super_sonic_wall_pressure[i] += 101325
  super_sonic_wall_pressure[i] /= 1000
  u_supersonic[i] /= 1000
  sub_sonic_pitot_average[i] += 101325
  sub_sonic_pitot_average[i] /= 1000
  pitot_subsonic[i] /= 1000
  super_sonic_pitot_average[i] += 101325
  super_sonic_pitot_average[i] /= 1000
  pitot_supersonic[i] /= 1000

print(u_supersonic)

import matplotlib.pyplot as plt
import numpy as np

locations = [24.7, 36.6, 48.3, 61, 73.7, 86.4, 99.1]

plt.errorbar(locations, sub_sonic_wall_pressure, yerr=u_subsonic, label='Subsonic Static Pressure', fmt='-o', capsize=5)
plt.xlabel('Location (mm)')
plt.ylabel('Pressure (kPa)')
plt.title('Subsonic Wall Pressure vs Location')
plt.legend()
plt.show()

plt.errorbar(locations, super_sonic_wall_pressure, yerr=u_supersonic, label='Supersonic Static Pressure', fmt='-o', capsize=5)
plt.xlabel('Location (mm)')
plt.ylabel('Pressure (kPa)')
plt.title('Supersonic Wall Pressure vs Location')
plt.legend()
plt.show()

plt.errorbar(locations, sub_sonic_pitot_average, yerr=pitot_subsonic, label='Subsonic Total Pressure', fmt='-o', capsize=5)
plt.xlabel('Location (mm)')
plt.ylabel('Pressure (kPa)')
plt.title('Subsonic Stagnation Pressure vs Location')
plt.legend()
plt.show()


plt.errorbar(locations, super_sonic_pitot_average, yerr=pitot_supersonic, label='Supersonic Total Pressure', fmt='-o', capsize=5)
plt.xlabel('Location (mm)')
plt.ylabel('Pressure (kPa)')
plt.title('Supersonic Stagnation Pressure vs Location')
plt.legend()
plt.show()

