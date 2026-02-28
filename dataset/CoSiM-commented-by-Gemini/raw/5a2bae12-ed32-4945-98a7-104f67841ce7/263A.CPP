/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem from Codeforces (263A).
 * @details The problem asks for the minimum number of moves (swapping adjacent
 * rows or columns) required to move a single '1' in a 5x5 matrix to the center cell.
 *
 * @algorithm
 * The minimum number of moves is equivalent to the Manhattan distance between the
 * current position of the '1' and the center of the matrix. The center of a 5x5
 * grid (using 0-based indexing) is at coordinates (2, 2).
 *
 * 1. The code iterates through the 5x5 matrix, reading each element.
 * 2. When the element '1' is found, its row and column indices (`loc1`, `loc2`) are stored.
 * 3. The Manhattan distance is then calculated using the formula:
 *    `|x1 - x2| + |y1 - y2|`.
 *    In this case, it's `abs(loc1 - 2) + abs(loc2 - 2)`.
 * 4. This distance is printed as the final answer.
 *
 * @complexity
 * Time Complexity: O(1), as the matrix size is fixed at 5x5. The loops run a
 * constant number of times (25).
 * Space Complexity: O(1), for the 5x5 matrix storage.
 */
#include <bits/stdc++.h>
#include <string.h>
using namespace std;
// --- Competitive Programming Macros ---
#define IOS                                                                    \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define endl "\n"
#define Max(x, y, z) max(x, max(y, z))
#define Min(x, y, z) min(x, min(y, z))
// Redefines `int` to `long long` to prevent potential integer overflows.
#define int long long
#define trace1(x) cerr << #x << ": " << x << endl
#define trace2(x, y) cerr << #x << ": " << x << " | " << #y << ": " << y << endl
#define trace3(x, y, z)                                                        \
  cerr << #x << ":" << x << " | " << #y << ": " << y << " | " << #z << ": "    \
       << z << endl

int32_t main() {
  IOS;
  /**
   * @variable a: A 5x5 matrix to store the input grid.
   * @variable loc1: Stores the row index of the '1'.
   * @variable loc2: Stores the column index of the '1'.
   */
  int a[5][5], loc1, loc2;
  
  /**
   * @block
   * @brief Reads the 5x5 matrix and finds the coordinates of the '1'.
   */
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      cin >> a[i][j];
      // Pre-condition: If the current element is 1, store its location.
      if (a[i][j] == 1) {
        loc1 = i;
        loc2 = j;
      }
    }
  }
  
  // Calculate and print the Manhattan distance to the center (2, 2).
  // This represents the minimum number of row/column swaps needed.
  cout << (abs(loc2 - 2) + abs(loc1 - 2));
  return 0;
}