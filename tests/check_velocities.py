import h5py
import numpy as np

f = h5py.File('results/test/two_particle_test.h5', 'r')

print("Initial state (t=0):")
print("="*70)

vels = f['timeseries/velocities'][0]
masses = f['timeseries/masses'][0]
pos = f['timeseries/positions'][0]

print("Velocities [fraction of c] and momentum [M_sun·c]:")
total_p = np.zeros(3)
for i in range(3):
    p = masses[i] * vels[i]
    total_p += p
    r = np.linalg.norm(pos[i])
    print(f"\nParticle {i} (m={masses[i]:.6e} M_sun, r={r/1e9:.6f} Gly):")
    print(f"  v = ({vels[i,0]:.6e}, {vels[i,1]:.6e}, {vels[i,2]:.6e})")
    print(f"  p = ({p[0]:.6e}, {p[1]:.6e}, {p[2]:.6e})")

print(f"\nTotal momentum: ({total_p[0]:.6e}, {total_p[1]:.6e}, {total_p[2]:.6e})")
print(f"|p_total| = {np.linalg.norm(total_p):.6e} M_sun·c")

f.close()
