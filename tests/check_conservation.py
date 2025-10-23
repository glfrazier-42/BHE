import h5py
import numpy as np

f = h5py.File('results/test/two_particle_test.h5', 'r')
cons = f['conservation']

print("Conservation errors over time:")
print("="*70)
print(f"{'Time [Gyr]':<15} {'Energy Error':<20} {'Momentum Error':<20}")
print("="*70)

times = f['timeseries/time'][:] / 1e9
energy_errors = cons['energy_error'][:]
momentum_errors = cons['momentum_error'][:]

for i in range(len(times)):
    print(f"{times[i]:<15.6f} {energy_errors[i]:<20.6e} {momentum_errors[i]:<20.6e}")

print("\nTotal momentum vectors:")
print("="*70)
momentum_vecs = cons['total_momentum'][:]
for i in range(min(5, len(times))):
    p = momentum_vecs[i]
    p_mag = np.linalg.norm(p)
    print(f"t={times[i]:.6f} Gyr: p=({p[0]:.6e}, {p[1]:.6e}, {p[2]:.6e}) M_sun·c, |p|={p_mag:.6e}")

f.close()
