import h5py
import numpy as np

f = h5py.File('results/test/two_particle_test.h5', 'r')
ts = f['timeseries']

print('Times [Gyr]:', ts['time'][:] / 1e9)
print('\nPositions at each timestep:')

for i in range(min(3, len(ts['time'][:]))):  # First 3 timesteps
    t_val = ts['time'][i] / 1e9
    print(f'\nTimestep {i} (t = {t_val:.6f} Gyr):')
    pos = ts['positions'][i]
    accreted = ts['accreted'][i]

    for j in range(3):
        r = np.linalg.norm(pos[j])
        acc_str = " [ACCRETED]" if accreted[j] else ""
        print(f'  Particle {j}: r={r/1e9:.3f} Gly{acc_str}')
        print(f'    Position: ({pos[j,0]/1e9:.6f}, {pos[j,1]/1e9:.6f}, {pos[j,2]/1e9:.6f}) Gly')

f.close()
