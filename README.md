<a id="toc"></a>
Table of Contents
=================
  * [Practice](#practice)
  * [resize image](#resize-image)
  * [Hide code](#hide-code)
    * [collapsible markdown?](#collapsible-markdown?)
  * [Color in github](#color-in-github)

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

# Hide code using \<details\> and \<summary\>
<details>
 
<summary>Code below is hidden</summary>
<code style="white-space:nowrap;"> import numpy as np </code>

</details>

[Go to TOC :arrow_heading_up:](#toc)
## collapsible markdown using \<details\> and \<summary\>

<details><summary>Answer</summary>
<p>

Everything between tag 'p' is hidden.

```python
print("hello world!")
```

</p>
</details>

[Go to TOC :arrow_heading_up:](#toc)
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


<h1 align="center">Hi 👋, I'm Bhishan Poudel</h1>
<h3 align="center">I am a data scientist with Ph.D degree in Astrophysics.</h3>

<p align="left"> <img src="https://komarev.com/ghpvc/?username=bhishanpdl" alt="bhishanpdl" /> </p>

- 🔭 I’m currently working on **janatahack sentiment analysis**

- 👨‍💻 All of my projects are available at [https://bhishanpdl.github.io/](https://bhishanpdl.github.io/)

- 📫 How to reach me **bhishanpdl@gmail.com**

<p align="left"><img src="https://devicons.github.io/devicon/devicon.git/icons/amazonwebservices/amazonwebservices-original-wordmark.svg" alt="aws" width="40" height="40"/> <img src="https://www.vectorlogo.zone/logos/gnu_bash/gnu_bash-icon.svg" alt="bash" width="40" height="40"/> <img src="https://devicons.github.io/devicon/devicon.git/icons/cplusplus/cplusplus-original.svg" alt="cplusplus" width="40" height="40"/> <img src="https://devicons.github.io/devicon/devicon.git/icons/docker/docker-original-wordmark.svg" alt="docker" width="40" height="40"/> <img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="git" width="40" height="40"/> <img src="https://www.vectorlogo.zone/logos/apache_hive/apache_hive-icon.svg" alt="hive" width="40" height="40"/> <img src="https://devicons.github.io/devicon/devicon.git/icons/linux/linux-original.svg" alt="linux" width="40" height="40"/> <img src="https://devicons.github.io/devicon/devicon.git/icons/mysql/mysql-original-wordmark.svg" alt="mysql" width="40" height="40"/> <img src="https://devicons.github.io/devicon/devicon.git/icons/postgresql/postgresql-original-wordmark.svg" alt="postgresql" width="40" height="40"/> <img src="https://devicons.github.io/devicon/devicon.git/icons/python/python-original.svg" alt="python" width="40" height="40"/></p><p><img align="left" src="https://github-readme-stats.vercel.app/api/top-langs/?username=bhishanpdl&layout=compact&hide=html" alt="bhishanpdl" /></p>

<p>&nbsp;<img align="center" src="https://github-readme-stats.vercel.app/api?username=bhishanpdl&show_icons=true" alt="bhishanpdl" /></p>

<p align="center">
<a href="https://twitter.com/bhishan_poudel" target="blank"><img align="center" src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/twitter.svg" alt="bhishan_poudel" height="30" width="30" /></a>
<a href="https://linkedin.com/in/bhishan poudel" target="blank"><img align="center" src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/linkedin.svg" alt="bhishan poudel" height="30" width="30" /></a>
<a href="https://stackoverflow.com/users/5200329" target="blank"><img align="center" src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/stackoverflow.svg" alt="5200329" height="30" width="30" /></a>
</p>

# Test
[Go to TOC :arrow_heading_up:](#toc)

# Mermaid Diagrams

### **Decision Tree for Choosing the Right Statistical Test**

```mermaid
flowchart TD
A[Goal: Compare Two Groups] --> B{Data Normal?<br>Shapiro-Wilk};
B -- Yes --> C{Variances Equal?<br>Levene's Test};
B -- No --> D[Use Non-Parametric<br>Mann-Whitney U Test];
C -- Yes --> E[Use Standard<br>Independent T-Test];
C -- No --> F[Use Robust Parametric<br>Welch's T-Test];

G[Goal: Compare >2 Groups] --> H{Data Normal?};
H -- Yes --> I{Variances Equal?};
H -- No --> J[Use Non-Parametric<br>Kruskal-Wallis H Test];
I -- Yes --> K[Use Standard<br>One-Way ANOVA];
I -- No --> L[Use Robust Parametric<br>Welch's ANOVA];

M[Goal: Paired Measurements<br>e.g., Pre-Post Treatment] --> N{Data Normal?};
N -- Yes --> O[Use Paired T-Test];
N -- No --> P[Use Non-Parametric<br>Wilcoxon Signed-Rank Test];
```
