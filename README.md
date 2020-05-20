# Practice
$$
bf = \frac{F_b + F_d} {F_b * f_{ratb} + F_d * f_{ratd}} * f_{ratb}
$$
$$
df = \frac{F_b + F_d} {F_b * f_{ratb} + F_d * f_{ratd}} * f_{ratd} 
$$
```
Here,
For file0: Fb is sum of pixels of bulge0.fits
For bulge: f_ratb = fr / (1+fr)
For disk : f_ratd = 1 / (1+fr)
Where,
fr = (sum_Fsb/sum_Fsd) / NUM_GALS
   = sum_of_pixels_fo_all_scaled_bulge / sum_of_pixels_fo_all_scaled_disk / NUM_GALS
```

# resize image
<img src="https://github.com/bhishanpdl/Tutorials_and_Lessons/blob/master/Tutorial_PySpark/images/pandas_vs_pyspark/multiple_cond.png" alt="alt text" width="400" height="400">

# Hide code
<details>
<summary>Summary text.</summary>
<code style="white-space:nowrap;">Hello World, how is it going?</code>
</details>
