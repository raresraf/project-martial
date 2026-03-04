/**
 * @file 69A.CPP
 * @brief Solution for Codeforces Problem 69A - "Young Physicist".
 *
 * This program checks for static equilibrium in a system of 3D force vectors.
 * It reads 'N' vectors and calculates the net force by summing the components
 * along each axis. If the net force is zero in all directions (i.e., the
 * resultant vector is the zero vector), the system is in equilibrium.
 *
 * @algorithm Vector component summation.
 * @time_complexity O(N), where N is the number of vectors.
 * @space_complexity O(1).
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#define fi first
#define se second
#define pb push_back
#define mp make_pair
#define pi 2 * acos(0.0)
#define eps 1e-9
#define PII pair<int, int>
#define PDD pair<double, double>
#define LL long long

using namespace std;

int N, x, y, z;
// fx, fy, fz represent the net force along the x, y, and z axes.
int fx, fy, fz;

int main() {
  scanf("%d", &N);
  fx = fy = fz = 0;
  /**
   * @brief Reads each vector and accumulates the component sums.
   * This loop iterates 'N' times, reading a 3D vector (x, y, z) and adding
   * its components to the running totals fx, fy, and fz.
   */
  while (N--) {
    scanf("%d %d %d", &x, &y, &z);
    fx += x, fy += y, fz += z;
  }
  /**
   * @brief Checks for equilibrium.
   * If the sum of forces along all three axes is zero, the body is in
   * equilibrium.
   */
  if (!fx && !fy && !fz)
    printf("YES\n");
  else
    printf("NO\n");
  return 0;
}