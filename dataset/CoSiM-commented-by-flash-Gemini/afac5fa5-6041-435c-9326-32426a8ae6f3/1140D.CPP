/**
 * @file 1140D.CPP
 * @brief Solves the "Minimum Triangulation" problem from Codeforces (1140D).
 * @details For a regular n-sided polygon with vertices labeled 1 to n, the problem
 * asks for the minimum cost to triangulate it. The cost of a triangle is the
 * product of its vertex labels.
 *
 * @algorithm
 * The optimal strategy is to form a "fan" triangulation from a single vertex.
 * For example, using vertex 1 as the common point, we can form triangles
 * (1, 2, 3), (1, 3, 4), ..., (1, n-1, n).
 * The total cost is the sum of the products of the vertex labels for each triangle:
 * Cost = (1 * 2 * 3) + (1 * 3 * 4) + ... + (1 * (n-1) * n).
 * This can be written as the summation:
 *   Sum_{i=2}^{n-1} (1 * i * (i+1))
 *
 * This code calculates this sum using a closed-form formula instead of a loop,
 * achieving an O(1) solution. The formula is derived from the sum of squares
 * and the sum of integers.
 *   Sum_{i=2}^{n-1} (i^2 + i) = (Sum_{i=1}^{n-1} (i^2 + i)) - (1^2 + 1)
 *   = (Sum_{i=1}^{n-1} i^2) + (Sum_{i=1}^{n-1} i) - 2
 *   = [(n-1)n(2n-1)/6] + [(n-1)n/2] - 2
 *
 * @time_complexity O(1), as it uses a direct mathematical formula.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>

int main() {
  int n;
  scanf("%d", &n);

  unsigned long long sum = 0;

  // The comment below is slightly misleading. The formula actually calculates
  // the cost for a fan triangulation from vertex 1, i.e., sum of (1*i*(i+1)).
  // split the regular polygon across vertex 1 into triangles
  // as 1-2-3, 1-3-4, 1-4-5 etc

  // O(1) calculation of Sum_{i=2}^{n-1} (i * (i+1))
  // Sum of squares from 1 to n-1
  sum += (unsigned long long)(n - 1) * n * (2 * n - 1) / 6;
  // Sum of integers from 1 to n-1
  sum += (unsigned long long)n * (n - 1) / 2;
  // Subtract the terms for i=1, which is (1*1 + 1) = 2.
  sum -= 2;

  printf("%llu\n", sum);
  return 0;
}