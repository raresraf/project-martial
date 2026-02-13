/**
 * @file 263A.CPP
 * @brief Checks a grid for a neighbor parity condition.
 * @details This program reads an n x n grid composed of 'x' and 'o' characters.
 * It then determines if every cell in the grid has an even number of adjacent 'o's.
 * The direct neighbors (up, down, left, right) are considered.
 *
 * @note The filename `263A.CPP` typically corresponds to the "Beautiful Matrix"
 * problem on Codeforces, but the code implemented here solves a different problem.
 * This code also uses a Variable Length Array (VLA) `a[n+2][n+2]`, which is a
 * non-standard C++ extension.
 *
 * @section algorithm Algorithm
 * 1. Read the grid size, `n`.
 * 2. Create an `(n+2)x(n+2)` integer grid `a`, padded with zeros to simplify boundary checks.
 * 3. Read the `n x n` character grid. For each cell:
 *    - Store `1` in grid `a` if the character is 'x'.
 *    - Store `-1` if the character is anything else (implicitly 'o').
 *    (The mapping is done to a 1-indexed subgrid within `a`).
 * 4. Iterate through each cell of the main `n x n` grid (from `i=1` to `n`, `j=1` to `n`):
 *    a. For the current cell, count how many of its four direct neighbors have the value `-1`.
 *    b. Check if this count is odd.
 *    c. If the count is odd for any cell, the condition is not met. Print "NO" and terminate.
 * 5. If the loops complete without finding any cell with an odd neighbor count, it means the
 *    condition holds for the entire grid. Print "YES".
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N^2), for iterating through the N x N grid.
 * - Space Complexity: O(N^2), to store the integer grid.
 */
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef vector<int> vi;
typedef pair<int, int> pii;
#define pb push_back
#define mp make_pair
#define ff first
#define ss second
#define sz(a) (int)a.size()
#define all(c) c.begin(), c.end()
#define rep(i, n) for (int i = 0; i < (n); i++)
#define rept(i, a, b) for (int i = (a); i < (b); i++)
#define fill(a, v) memset(a, v, sizeof(a))
#define foreach(c, it)                                                         \
  for (__typeof(c.begin()) it = c.begin(); it != c.end(); ++it)

int main() {
  int n;
  cin >> n;
  // Using a Variable Length Array (VLA), a non-standard C++ extension.
  // The array is padded by 2 to simplify neighbor checking at the boundaries.
  int a[n + 2][n + 2];
  fill(a, 0);
  char x;
  // Read the grid character by character.
  rep(i, n) {
    rep(j, n) {
      cin >> x;
      // Map 'x' to 1 and other characters ('o') to -1 for easier checking.
      if (x == 'x') {
        a[i + 1][j + 1] = 1;
      } else
        a[i + 1][j + 1] = -1;
    }
  }
  int flag = 1; // Flag to indicate if the condition holds true.
  // Iterate through each cell of the grid (excluding the padding).
  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= n; j++) {
      int count = 0;
      // Count how many direct neighbors are 'o' (represented by -1).
      if (a[i][j + 1] == -1)
        count++;
      if (a[i][j - 1] == -1)
        count++;
      if (a[i + 1][j] == -1)
        count++;
      if (a[i - 1][j] == -1)
        count++;

      // If the number of 'o' neighbors is not even, the condition is violated.
      if (count % 2 == 0)
        continue;
      else {
        flag = 0;
        cout << "NO\n";
        break;
      }
    }
    if (!flag)
      break;
  }
  // If the flag remains 1, all cells have an even number of 'o' neighbors.
  if (flag)
    cout << "YES\n";

  return 0;
}
