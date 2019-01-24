# Algebra

Linear equation:
$$
\boldsymbol { Y } = \boldsymbol { A } \boldsymbol { X }
$$

Dependent variable Y is called response or target:
$$
\boldsymbol { Y } = \left[ \begin{array} { c } { y _ { 1 } } \\ { y _ { 2 } } \\ { \cdots } \\ { y _ { N } } \end{array} \right]
$$

Independent variable, A is augmented matrix.
$$
\boldsymbol { A } = \left[ \begin{array} { c c } { 1 } & { x _ { 1 } } \\ { 1 } & { x _ { 2 } } \\ { \cdots } \\ { 1 } & { x _ { N } } \end{array} \right]
$$


Covariance matrix:
$$
C = \left[ \begin{array} { c c c c } { \sigma _ { y 1 } ^ { 2 } } & { 0 } & { \cdots } & { 0 } \\ { 0 } & { \sigma _ { y 2 } ^ { 2 } } & { \cdots } & { 0 } \\ { 0 } & { y _ { y 2 } } & { \cdots } & { 0 } \\ { 0 } & { 0 } & { \cdots } & { \sigma _ { y N } ^ { 2 } } \end{array} \right]
$$

Covariance matrix when we have uncertainties in both x and y:
$$
\left[ \begin{array} { c c } { \sigma _ { x } ^ { 2 } } & { \rho _ { x y } \sigma _ { x } \sigma _ { y } } \\ { \rho _ { x y } \sigma _ { x } \sigma _ { y } } & { \sigma _ { y } ^ { 2 } } \end{array} \right]
$$

Solution to linear equation:
$$
parameters = \left[ \begin{array} { c } { \mathrm { b } } \\ { \mathrm { m } } \end{array} \right] = \boldsymbol { X } = \left[ \boldsymbol { A } ^ { \top } \boldsymbol { C } ^ { - 1 } \boldsymbol { A } \right] ^ { - 1 } \left[ \boldsymbol { A } ^ { \top } \boldsymbol { C } ^ { - 1 } \boldsymbol { Y } \right]
$$
