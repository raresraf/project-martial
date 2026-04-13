/**
 * @file 263A.CPP
 * @brief This program finds the minimum number of moves to bring the element '1'
 *        to the center of a 5x5 matrix. The center is considered to be at
 *        coordinates (3,3) (1-indexed). The movement is restricted to adjacent
 *        cells (up, down, left, right).
 * Algorithm: Manhattan Distance Calculation.
 * Time Complexity: O(1) as the matrix size is fixed at 5x5.
 * Space Complexity: O(1) for storing the 5x5 matrix and a pair of coordinates.
 */
#include <bits/stdc++.h>
#include <iostream>
using namespace std;
typedef long long int ll;
typedef long double ld;
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

//  112-A Petya String
int main() {
  int a[5][5];
  pair<int, int> x;
  /**
   * Block Logic: Reads the 5x5 matrix from standard input and identifies the
   *              coordinates of the element with value '1'.
   * Invariant: After each iteration of the outer loop, a row of the matrix has
   *            been read. After the nested loops, the 'x' pair holds the 1-indexed
   *            coordinates of the element '1'.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1)
        // Inline: Stores the 1-indexed coordinates of '1'.
        x = make_pair(i + 1, j + 1);
    }
  }
  // Calculates the Manhattan distance from the current position of '1' to the center (3,3).
  int fc = abs(3 - x.first) + abs(3 - x.second);
  cout << fc << endl;

  return 0;
}