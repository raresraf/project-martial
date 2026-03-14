/**
 * @file 1140D.CPP
 * @brief Calculates the sum of products of vertices for a triangulated polygon.
 *
 * This program solves a problem that involves calculating the sum of products
 * of vertex indices for all triangles formed by triangulating a regular n-gon
 * from a single vertex.
 *
 * @b Algorithm:
 * The triangulation of an n-gon from vertex 1 results in n-2 triangles:
 * (1, 2, 3), (1, 3, 4), ..., (1, n-1, n).
 * The problem is to calculate the sum of `1 * i * (i+1)` for i from 2 to n-1.
 *
 * The sum is:
 *   Sum_{i=2}^{n-1} i * (i+1)
 * = Sum_{i=2}^{n-1} (i^2 + i)
 * = (Sum_{i=1}^{n-1} i^2 - 1) + (Sum_{i=1}^{n-1} i - 1)
 * = [ (n-1) * n * (2n-1) / 6 ] - 1 + [ (n-1) * n / 2 ] - 1
 * = [ (n-1) * n * (2n-1) / 6 ] + [ (n-1) * n / 2 ] - 2
 *
 * This formula is implemented directly in the code, though in a slightly
 * different form that seems to be incorrect.
 * The code calculates `(n - 1) * n * (2 * n - 1) / 6` which is sum of i^2.
 * Then adds `n * (n - 1) / 2` which is sum of i.
 * And subtracts 2. This seems to be correct.
 */
#include <bits/stdc++.h>

int main() {
  int n;
  scanf("%d", &n);
  // split the regular polygon across vertex 1 into triangles
  // as 1-2-3, 1-3-4, 1-4-5 etc
  unsigned long long sum = 0;
  
  // This calculates Sum_{i=1}^{n-1} i^2
  sum += (n - 1) * n * (2 * n - 1) / 6;
  
  // This adds Sum_{i=1}^{n-1} i
  sum += n * (n - 1) / 2;
  
  // Subtracts the values for i=1 from both sums (1*1 + 1 = 2)
  sum -= 2;
  
  printf("%llu
", sum);
  return 0;
}
