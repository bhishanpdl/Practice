$$
\subsection{Creation of Scaled Bulge, Disk, and Monochromatic Images}
We have total 201 number of HST images, so we have 201 bulge images and 201 disk images.
From these two folders we create so called scaled\_bulge, scaled\_disk, and scaled\_bulge\_disk folders. For this,
we first find the bulge\_factor (bf) and disk\_factor (df) then we create scaled galaxies.

 \begin{eqnarray}
 scaled\_bulge = bf * bulge.fits \\
 scaled\_disk = df * disk.fits
 \end{eqnarray}
 
 

To find bulge and disk factors, first we find fraction for bulge ratio and fraction of disk ratio as follows:

 \begin{eqnarray}
 f_{ratb} = \frac{\int_{\lambda0}^{\lambda20} f_{bz}(\lambda)d\lambda}
 {\int_{\lambda{hst0}}^{\lambda_{hst20}} f_{bzcut}(\lambda)d\lambda} \\
 f_{ratd} = \frac{\int_{\lambda0}^{\lambda20} f_{dz}(\lambda)d\lambda}
 {\int_{\lambda{hst0}}^{\lambda_{hst20}} f_{dzcut}(\lambda)d\lambda}
 \end{eqnarray}
Here, $f_{bz}$ is the flux column from the SED file according the redshift $z$ for the bulge and $f_{bzcut}$ is the 
flux column for cutout galaxy. Here, we have used the galaxy cutout redshift as $ z_{cutout} = 0.2$. Similarly we have the flux columns for disk galaxies.

The wavelengths $\lambda_0$ and $\lambda_{20}$ are the LSST R-band filter blue and red wavelengths. This range is $5520 \AA$ to $6910 \AA$ (Refer to: $https://www.lsst.org/about/camera/features$).
We divide these wavelengths by a factor ($1 + z$) to get the range 2208 to 2764 for the redshift of 1.5.

Similarly, for the HST the wavelengths are $\lambda_{hst0} = 7077.5 \AA$ and $\lambda_{hst0} = 9588.5 \AA$ after dividing by $ 1 + z = 1.2$ we get $\lambda_{hst0} = 5897.9 \AA$ and $\lambda_{hst0} = 7990.4 \AA$. We can get more details about HST ACS/WFC filter at the website $http://www.stsci.edu/hst/acs/documents/handbooks/current/c05_imaging2.html$.

Then, we get bulge factor and disk factor using the formula:
 \begin{eqnarray}
 bf = \frac{F_b + F_d} {F_b * f_{ratb} + F_d * f_{ratd}} * f_{ratb} \\
 bd = \frac{F_b + F_d} {F_b * f_{ratb} + F_d * f_{ratd}} * f_{ratd} \\
 \end{eqnarray}
 
 where, $F_b$ is the flux of a bulge file (e.g. $simdatabase/bulge_f8/f814w_bulge0.fits$) and $F_d$ is the flux of a disk file (e.g. $simdatabase/disk_f8/f814w_disk0.fits$) for 201 bulge and disk files we have 201 bulge and disk factors.
 
After we get these bulge and disk factors we simply multiply them by the bulge.fits and disk.fits to get scaled\_bulge.fits and scaled\_disk.fits.
$$