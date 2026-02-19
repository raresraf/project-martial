/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem using 0-indexed coordinates.
 * @details This program calculates the minimum number of moves to shift a single '1'
 * in a 5x5 matrix to the center. A move consists of swapping with an adjacent cell.
 * The solution uses 0-indexing for rows and columns, placing the center at (2, 2).
 *
 * @algorithm The program iterates through the entire 5x5 matrix to locate the '1'.
 * Once its 0-indexed coordinates (row, col) are found, the Manhattan distance to the
 * center (2, 2) is calculated as `|2 - row| + |2 - col|`. This distance represents
 * the minimum number of single-step moves required.
 *
 * @time_complexity O(1), as the program performs a fixed number of operations (25 reads)
 * regardless of the '1's position.
 * @space_complexity O(1), as it uses a constant amount of memory.
 */
/*
 ~ Author    : @tridib_2003
*/

#include <bits/stdc++.h>
using namespace std;

// Common macros for competitive programming, though most are unused in this specific problem.
#define ll long long
#define PI 3.1415926535897932384626
#define MOD 1000000007
#define vi vector<int>
#define mk(arr, n, type) type *arr = new type[n];
#define FOR(a, c) for (int(a) = 0; (a) < (c); (a)++)
#define w(x)                                                                   \
  int x;                                                                       \
  cin >> x;                                                                    \
  while (x--)
// Macro to optimize C++ stream I/O.
#define FIO                                                                    \
  ios::sync_with_stdio(false);                                                 \
  cin.tie(0);                                                                  \
  cout.tie(0);

int main() {

  // FIO macro call to optimize I/O operations.
  FIO;

  int rowIdx = 0, colIdx = 0;

  /**
   * Block Logic: Iterate through the 5x5 matrix to find the 0-indexed coordinates of the '1'.
   * The coordinates are stored in `rowIdx` and `colIdx`.
   */
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {

      int val;
      cin >> val;

      if (val == 1) {
        rowIdx = i;
        colIdx = j;
      }
    }
  }

  // Block Logic: Calculate the Manhattan distance from the '1's position to the center (2, 2).
  int moves = abs(2 - rowIdx) + abs(2 - colIdx);
  cout << moves << '\n';

  return 0;
}