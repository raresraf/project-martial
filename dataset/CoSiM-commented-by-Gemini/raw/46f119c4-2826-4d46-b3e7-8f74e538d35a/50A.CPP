/**
 * @file raw/46f119c4-2826-4d46-b3e7-8f74e538d35a/50A.CPP
 * @brief Solution to Codeforces problem 50A - Domino piling.
 *
 * @algorithm
 * The problem is to find the maximum number of 2x1 dominoes that can be placed
 * on a rectangular board of size M x N without overlapping.
 *
 * The total area of the board is `M * N`.
 * Each domino covers an area of 2.
 *
 * The maximum number of dominoes is the total area of the board divided by the
 * area of a single domino. Since we cannot use partial dominoes, we take the
 * integer part of the division.
 *
 *   Maximum dominoes = floor((M * N) / 2)
 *
 * In C++, integer division `(m * n) / 2` automatically performs this floor
 * operation for positive integers.
 *
 * The algorithm is as follows:
 * 1. Read the board dimensions, M and N.
 * 2. Calculate `(M * N) / 2` using integer division.
 * 3. Print the result.
 *
 * @complexity
 * - Time Complexity: O(1), as the solution involves a few arithmetic operations.
 * - Space Complexity: O(1), using only a few variables to store the dimensions.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // n, m: dimensions of the rectangular board.
  int n, m;
  cin >> m >> n;
  
  // Calculate and print the maximum number of dominoes.
  // This is the total area of the board integer-divided by the area of one domino (2).
  cout << (m * n) / 2;
  return 0;
}
