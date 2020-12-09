"""
Implementation of a Complementary Filter using accelerometer and gyro data
collected from a smartphone.

Refer to the following videos of Brian Douglas for the Control theory:
- https://www.youtube.com/watch?v=whSw42XddsU
- https://youtu.be/nkq4WkX7CFU?t=732

"""


# External imports
import pandas as pd
import numpy as np
from numpy import rad2deg as r2d
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

# Local imports
from utils import load_data
from config import FILENAME


###############################################################################
# Data preparation
###############################################################################

# Load data
df_acc = pd.read_csv('data4_accm.csv')
df_gyr = pd.read_csv('data4_gyrm.csv')

# Load time and make it start at zero
t_acc = df_acc['time'] - df_acc.iloc[0, 0]
t_gyr = df_gyr['time'] - df_gyr.iloc[0, 0]

# Prepare interpolation objetcs to get an "accurate" sampling rate
# Note: the accelerometer has a higher average sampling rate than the gyro, and
# the sampling frequency is not really constant for neither of them. So I'll 
# force the two of them to have the same consntant sampling rate by 
# interpolating their values and sampling at a desired constant frequency.
accel_x = interp1d(t_acc, df_acc['X_value'], kind='linear')
accel_y = interp1d(t_acc, df_acc['Y_value'], kind='linear')
accel_z = interp1d(t_acc, df_acc['Z_value'], kind='linear')
gyr_x   = interp1d(t_gyr, df_gyr['X_value'], kind='linear')
gyr_y   = interp1d(t_gyr, df_gyr['Y_value'], kind='linear')
gyr_z   = interp1d(t_gyr, df_gyr['Z_value'], kind='linear')

# Samping time
dt = 0.04

# Define the new time vector
t = np.arange(0, t_acc.iloc[-2], dt)

# Interpolate the new values based on the new time vector
accel_x = accel_x(t)
accel_y = accel_y(t)
accel_z = accel_z(t)

gyr_x = r2d(gyr_x(t))
gyr_y = r2d(gyr_y(t))
gyr_z = r2d(gyr_z(t))


# Initial value for a sample one step in the past
roll_k1 = 0

# Time constant of the low-pass filter
tau = 0.88

# Output vector
roll = np.zeros(len(t))


###############################################################################
# Complementary filter
###############################################################################

for i in range(len(t)):

    accel_angle = np.rad2deg(np.arctan2(accel_x[i], accel_z[i]))

    if i == 0:
        roll[i] = (1-tau) * accel_angle + tau * gyr_x[i] * dt + tau * roll_k1
    
    else:
        roll[i] = (1-tau) * accel_angle + tau * gyr_x[i] * dt + tau  * roll[i-1]


###############################################################################
# Plotting
###############################################################################

# Calculate the roll angle based on only accelerometer data
raw_accel_angle = np.rad2deg(np.arctan2(accel_x, accel_z))

# Calculate the roll angle based only on gyro data
raw_inte_gyro = cumtrapz(gyr_x, dx=dt, initial=gyr_x[0])

fig,ax = plt.subplots(1, 1, sharex=True)

ax.plot(t, roll,            label='Complementary filter')
ax.plot(t, raw_accel_angle, label='Raw accel angle')
ax.plot(t, raw_inte_gyro,   label='Integrated raw gyro')
plt.legend()

plt.show()