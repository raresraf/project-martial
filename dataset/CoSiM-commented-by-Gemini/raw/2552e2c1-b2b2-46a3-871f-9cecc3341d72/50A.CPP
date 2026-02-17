/**
 * @file 50A.CPP
 * @brief Solution for Codeforces Problem 50A, "Domino piling".
 * @details This program calculates the maximum number of 2x1 dominoes that can be placed
 * on a rectangular board of size M x N.
 *
 * @algorithm Direct calculation using an area-based formula. The total area of the
 * board is M * N. Since each domino covers an area of 2, the maximum number of
 * non-overlapping dominoes is the total area divided by 2. Integer division
 * inherently floors the result, which correctly handles boards with an odd total area.
 *
 * @time_complexity O(1), as the solution involves a single arithmetic operation.
 * @space_complexity O(1), as the memory usage is constant.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int M, N; // M and N are the dimensions of the rectangular board.
  cin >> M >> N;
  
  // Calculate the maximum number of dominoes. The total area of the board is M * N.
  // Each domino has an area of 2. Integer division handles the flooring implicitly.
  cout << M * N / 2 << endl;
  
  return 0;
}