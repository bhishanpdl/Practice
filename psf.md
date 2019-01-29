# PSF Creation for Bulge, Disk, and Monochromatic Images
From the PHOSIM Software we have created 21 narrowband PSFs. Now we will use them to create PSF for scaled bulge, disk, and monochromatic images. The scaled psf files are given by formula:

$$
p_b = \frac{b0 * p0 + b1 * p1 + ... + b20 * p20}{b0 + b1 + ... + b20}\\
p_d = \frac{d0 * p0 + d1 * p1 + ... + d20 * p20}{d0 + d1 + ... + d20}
$$

Here, $p_b$, $p_d$,and $p_m$ are psf for bulge, disk, and monochromatic respectively. Also the quantities $b0, b1, ..., b20$ and $d0, d1, ..., d20$ are bulge and disk weights for 21 narrowbands. 

These quantities are the integrated flux in the given narrowbands. 
For example, for LSST R band filter the blue and red wavelength range is 2208 to 2764 Angstrom. 
We divide this range into 21 parts and integrate the flux in that range to get the bulge and disk factor for 
that range using SED file for bulge and disk.
