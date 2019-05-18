#!/bin/env python3
from mpmath import *
from scipy.stats import poisson

def fp_std(m, n, k):
	return (1-((1-(1/m))**(k*n)))**k

def fp_blocked(m, n, k, B):
	y = B*(n/m)
	fp = nsum(lambda i: (poisson.ppf(i, y) * fp_std(B,i,k)), [0, inf])
	return fp

def internal_sum(S,z,i,k):
	return nsum(lambda j: (poisson.ppf(j, z) * fp_std(S,j,k/S) , [1, i]))

def fp_cache(m,n,k,B,S,z):
	y = B*(n/m)
	fp = nsum(lambda i: (poisson.ppf(i, y) * ((internal_sum(S,(S*((i*z)/B)),i,k))**z)), [0, inf])
	


def main():
	print(fp_std(2147483648, 536870912, 3))
	print(fp_blocked(2147483648, 536870912, 3, 512))
	print(fp_cache(2147483648, 536870912,3, 512, 1, 1))




if __name__ == '__main__':
	main()

