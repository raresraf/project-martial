/**
 * @file 50A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Domino piling").
 * The program calculates the maximum number of 2x1 dominoes that can be placed
 * on a rectangular M x N board.
 *
 * NOTE: This file contains several boilerplate macros that are not used in this
 * specific solution.
 *
 * @algorithm Mathematical Formula. The total area of the board is M * N.
 * Each domino covers an area of 2. Therefore, the maximum number of dominoes
 * is the total area divided by 2, rounding down. In integer arithmetic, this
 * is simply `(m * n) / 2`.
 *
 * @time_complexity O(1), as it's a direct calculation.
 * @space_complexity O(1).
 */
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
#define sz(a) int((a).size())
#define pb push_back
#define mp make_pair
#define all(c) (c).begin(), (c).end()
#define tr(c, i) for(typeof((c).begin() i = (c).begin(); i != (c).end(); i++)
#define present(c, x) ((c).find(x) != (c).end())
#define cpresent(c, x) (find(all(c), x) != (c).end())

using namespace std;

void solve() {
  int m, n, ans;
  cin >> m >> n;

  /**
   * @block
   * @brief Calculates the number of dominoes.
   * Although the simple formula `(m * n) / 2` works for all cases, this code
   * splits the logic.
   */
  if (n % 2 == 0 || m % 2 == 0) {
    // If either dimension is even, the board can be perfectly tiled.
    ans = (m * n) / 2;
  } else {
    // If both dimensions are odd, this is a more complex but equivalent way
    // of calculating `(m * n) / 2`. It considers tiling the largest even
    // subgrid and then filling the remaining row/column.
    ans = max(m * (n - 1) / 2 + m / 2, n * (m - 1) / 2 + n / 2);
  }
  cout << ans << "\n";
}

int main() {
  // Fast I/O setup
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  solve();
  return 0;
}