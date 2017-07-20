#!/usr/bin/env python3
# -*- coding: utf-8 -*-#
#
# Author      : Bhishan Poudel; Physics Graduate Student, Ohio University
# Date        : Jan 4, 2017
# Last update : Jun 19, 2017 Mon

def missing_galaxies():
    """Missing galaxies.

    e.g. f814w_devauc_two_comp1.fits
    """
    pth = '/Users/poudel/Research/galfit_usage/verify_2cmp_fitting/' +\
          'f814w_fitting/devauc_two_comp/f814w_devauc_two_comp*.fits'
    nums = natsort.natsorted([int(re.search('(.+?)(\d+)(\.fits$)', f).group(2))
                              for f in glob.glob(pth)])
    missing = [i for i in range(302) if i not in nums]
    print('Total number of missing galaxies    : ', len(missing))
    print('Total number of non-missing galaxies: ', 302 - len(missing))
    print('Missing galaxies: ', missing)

if __name__ == "__main__":
    
    # Imports
    import glob
    import re
    import natsort
    
    missing_galaxies()
