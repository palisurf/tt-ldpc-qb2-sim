#!/usr/bin/env python3
import numpy as np, math

theory_mu = 3.169786
theory_sigma = 2.517851
theory_ber = 0.5 * math.erfc(np.sqrt(0.5 * 10**(2.0/10.0)))
print(f"Theoretical Channel BER at 2.0 dB: {theory_ber:.6f} ({theory_ber*100:.2f}%)")

for snr in np.arange(1.0, 2.05, 0.05):
    es_n0 = 0.5 * 10**(snr/10.0)
    b = 0.5 * math.erfc(np.sqrt(es_n0))
    print(f"SNR={snr:4.2f} dB: Channel BER={b:7.5f} ({b*100:5.2f}%)")
