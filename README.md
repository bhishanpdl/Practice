# Practice
```
For file0: Fb is sum of pixels of bulge0.fits
For bulge: f_ratb = fr / (1+fr)
For disk : f_ratd = 1 / (1+fr)
Where,
fr = (sum_Fsb/sum_Fsd) / NUM_GALS
   = sum_of_pixels_fo_all_scaled_bulge / sum_of_pixels_fo_all_scaled_disk / NUM_GALS
```
$$
bf = \frac{F_b + F_d} {F_b * f_{ratb} + F_d * f_{ratd}} * f_{ratb}
$$
$$
df = \frac{F_b + F_d} {F_b * f_{ratb} + F_d * f_{ratd}} * f_{ratd} 
$$
