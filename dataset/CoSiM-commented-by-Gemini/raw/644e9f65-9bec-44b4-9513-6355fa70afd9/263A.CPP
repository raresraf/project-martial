/**
 * @file 263A.CPP
 * @brief Solution for the "Beautiful Matrix" problem.
 * @author Abhinav
 *
 * @problem A 5x5 matrix contains a single '1' and 24 '0's. The goal is to move
 * the '1' to the center of the matrix (row 2, column 2 in 0-based indexing)
 * using the minimum number of moves, where a move is an adjacent row or column swap.
 *
 * @algorithm The minimum number of moves is the Manhattan distance between the '1's
 * initial position and the center. For a 0-indexed grid, the center is (2, 2).
 * The algorithm finds the coordinates (x, y) of the '1' and then calculates
 * the distance as `|x - 2| + |y - 2|`.
 *
 * @time_complexity O(1), as the matrix size is fixed and small (5x5).
 * @space_complexity O(1).
 */

#include <bits/stdc++.h>

using namespace std;

// Type definitions for convenience.
typedef long long ll;
typedef vector<ll> vi;
typedef map<ll, ll> mllmp;

// Macro for fast I/O in competitive programming.
#define faster                                                                 \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(0);
#define make_pair mp

const ll mod = 1e9 + 7;

// Macro for a standard for-loop.
#define rep(i, j, k) for (ll i = j; i < k; i++)
// Macro for a reverse for-loop.
#define repv(i, j, k) for (ll i = j; i > k; i--)

int main() {
  faster;
  ll x, y; // To store the 0-indexed coordinates of the '1'.
  vector<vector<int>> v(5, vector<int>(5));
  
  // Read the 5x5 matrix and find the position of the '1'.
  rep(i, 0, 5) rep(j, 0, 5) {
    cin >> v[i][j];
    if (v[i][j] == 1) {
      x = i;
      y = j;
    }
  }
  
  // This block calculates the Manhattan distance |x - 2| + |y - 2| in a verbose way.
  if (x == 2 && y == 2)
    cout << "0"; // Already at the center.
  else if (x == 2)
    cout << abs(y - 2); // Only need to move columns.
  else if (y == 2)
    cout << abs(x - 2); // Only need to move rows.
  else
    // Need to move both rows and columns.
    cout << abs(y - 2) + abs(x - 2);
    
  return 0;
}
