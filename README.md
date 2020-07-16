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

## collapsible markdown?

<details><summary>CLICK ME</summary>
<p>

#### yes, even hidden code blocks!

```python
print("hello world!")
```

</p>
</details>

# Color in github
- ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `#f03c15`
- ![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) `#c5f015`
- ![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) `#1589F0`
- ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `red text`
- ![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) `green text`
- ![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) `blue text`

```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@

outside @@ text in purple (and bold)@@ outside
```

```diff
- text in red
```

```html
   <p style='color:red'>This is some red text if it is working .</p>
```


