# Creation of Scaled Bulge, Disk, and Monochromatic Images
We have total 201 number of HST images, so we have 201 bulge images and 201 disk images.
From these two folders we create so called `scaled_bulge`, `scaled_disk`, and `scaled_bulge_disk` folders. 
For this, we first find the `bulge_factor` (bf) and `disk_factor` (df) then we create scaled galaxies.

  $$scaledbulge = bf * bulge.fits$$
  $$ scaleddisk = df * disk.fits$$
 
 

To find bulge and disk factors, first we find fraction for bulge ratio and fraction of disk ratio as follows:

 $$
 f_{ratb} = \frac{\int_{\lambda0}^{\lambda20} f_{bz}(\lambda)d\lambda}
 {\int_{\lambda{hst0}}^{\lambda_{hst20}} f_{bzcut}(\lambda)d\lambda} \\
 f_{ratd} = \frac{\int_{\lambda0}^{\lambda20} f_{dz}(\lambda)d\lambda}
 {\int_{\lambda{hst0}}^{\lambda_{hst20}} f_{dzcut}(\lambda)d\lambda}
 $$
Here, $f_{bz}$ is the flux column from the SED file according the redshift $z$ for the bulge and $f_{bzcut}$ is the 
flux column for cutout galaxy. Here, we have used the galaxy cutout redshift as $ z_{cutout} = 0.2$. Similarly we have the flux columns for disk galaxies.

The wavelengths $\lambda_0$ and $\lambda_{20}$ are the LSST R-band filter blue and red wavelengths. This range is `5520 Angstrom` to `6910 Angstrom` [Refer to](https://www.lsst.org/about/camera/features)).
We divide these wavelengths by a factor ($1 + z$) to get the range 2208 to 2764 for the redshift of 1.5.

Similarly, for the HST the wavelengths are $\lambda_{hst0} = 7077.5 $ Angstrom and $\lambda_{hst0} = 9588.5 $ Angstrom after dividing by $ 1 + z = 1.2$ we get $\lambda_{hst0} = 5897.9 $ Angstrom and $\lambda_{hst0} = 7990.4 $ Angstrom. We can get more details about HST ACS/WFC filter at the [website](http://www.stsci.edu/hst/acs/documents/handbooks/current/c05_imaging2.html).

Then, we get bulge factor and disk factor using the formula:
 $$
 bf = \frac{F_b + F_d} {F_b * f_{ratb} + F_d * f_{ratd}} * f_{ratb} \\
 bd = \frac{F_b + F_d} {F_b * f_{ratb} + F_d * f_{ratd}} * f_{ratd}
$$
 
 where, `F_b` is the flux of a bulge file (e.g. `simdatabase/bulge_f8/f814w_bulge0.fits`) and `F_d` is the flux of a disk file (e.g. `simdatabase/disk_f8/f814w_disk0.fits`) for 201 bulge and disk files we have 201 bulge and disk factors.
 
After we get these bulge and disk factors we simply multiply them by the `bulge.fits` and `disk.fits` to get `scaled_bulge.fits` and `scaled_disk.fits`.

# PSF Creation for Bulge, Disk, and Monochromatic Images
From the PHOSIM Software we have created 21 narrowband PSFs. Now we will use them to create PSF for scaled bulge, disk, and monochromatic images. The scaled psf files are given by formula:
$$
p_b = \frac{b0*p0 + b1*p1 + ... + b20*p20}{b0 + b1 + ... + b20} \\
p_d = \frac{d0*p0 + d1*p1 + ... + d20*p20}{d0 + d1 + ... + d20} \\
p_m = f_{rd} \ p_d + f_{rb} \ p_b
$$
Here, $p_b$, $p_d$,and $p_m$ are psf for bulge, disk, and monochromatic respectively. Also the quantities $b0, b1, ..., b20$ and $d0, d1, ..., d20$ are bulge and disk weights for 21 narrowbands. These quantities are the integrated flux in the given narrowbands. For example, for LSST R band filter the blue and red wavelength range is 2208 to 2764 Angstrom. We divide this range into 21 parts and integrate the flux in that range to get the bulge and disk factor for that range using SED file for bulge and disk.
