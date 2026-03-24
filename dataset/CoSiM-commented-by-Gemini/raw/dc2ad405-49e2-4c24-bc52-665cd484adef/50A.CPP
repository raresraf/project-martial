/**
 * @file 50A.CPP
 * @brief Solution for Codeforces Problem 50A - "Domino piling".
 *
 * @details
 * This program calculates the maximum number of 2x1 dominoes that can be
 * placed on a rectangular board of size M x N.
 *
 * Algorithm:
 * The solution is a direct mathematical calculation. A single domino covers an
 * area of 2. The total area of the board is M * N. Therefore, the maximum
 * number of dominoes is the total area divided by 2.
 * Formula: floor((M * N) / 2)
 *
 * The implementation uses integer division, which for positive numbers
 * correctly computes the floor of the result.
 *
 * Time Complexity: O(1), as it involves a single arithmetic operation.
 * Space Complexity: O(1), using only a few variables for storage.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  // M and N are the dimensions of the board.
  int M, N, domino = 2;
  int result;
  cin >> M >> N;
  
  // Functional Utility: Calculate the maximum number of dominoes by dividing the
  // total board area by the area of a single domino.
  result = (M * N) / domino;
  
  cout << result << endl;
  return 0;
}
