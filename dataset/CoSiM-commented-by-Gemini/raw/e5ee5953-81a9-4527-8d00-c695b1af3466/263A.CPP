/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem from Codeforces (Problem 263A).
 * @details The program reads a 5x5 matrix containing a single '1' and all other
 * elements as '0'. It calculates the minimum number of single-step moves
 * (up, down, left, or right) required to move the '1' to the center of the
 * matrix, which is at position (3,3).
 *
 * @section
 * Algorithm:
 * 1. Read the elements of the 5x5 matrix.
 * 2. While reading, identify the 1-indexed coordinates (row, col) of the element '1'.
 * 3. Calculate the Manhattan distance between the '1's current coordinates (r, c)
 *    and the center coordinates (3, 3).
 * 4. The formula for Manhattan distance is |r2 - r1| + |c2 - c1|. Here, it is
 *    |3 - r| + |3 - c|.
 * 5. Print the calculated distance.
 *
 * @section
 * Complexity Analysis:
 * - Time Complexity: O(1), as the input size is a constant 5x5 grid.
 * - Space Complexity: O(1), for storing the 5x5 matrix.
 */
#include <bits/stdc++.h>
#include <iostream>
using namespace std;
typedef long long int ll;
typedef long double ld;

// Macros for common competitive programming operations.
#define mp make_pair
#define v vector
#define pb push_back
#define fi first
#define se second
#define ALL(v) v.begin(), v.end()
#define UN(v)                                                                  \
  sort((v).begin(), (v).end()), v.erase(unique(v.begin(), v.end()), v.end())
#define N 31622779
#define mod 998244353

// Solves 263-A, Beautiful Matrix
int main() {

  int a[5][5];
  pair<int, int> x;
  /**
   * Block Logic: Reads the 5x5 matrix and finds the coordinates of the '1'.
   * Pre-condition: The input is a 5x5 grid of integers.
   * Post-condition: The pair 'x' holds the 1-indexed coordinates of the '1'.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1)
        x = make_pair(i + 1, j + 1);
    }
  }
  // Algorithm: Calculate the Manhattan distance from the position of '1' to the center (3,3).
  // This represents the minimum number of swaps (moves) required.
  int fc = abs(3 - x.first) + abs(3 - x.second);
  cout << fc << endl;

  return 0;
}