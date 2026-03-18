/**
 * @file 50A.CPP
 * @brief This program calculates the maximum number of 2x1 dominoes that can be placed on an M x N chessboard.
 * Algorithm: Mathematical calculation based on the dimensions of the chessboard.
 * Time Complexity: O(1) as it involves direct arithmetic operations.
 * Space Complexity: O(1) as only a few variables are used.
 */
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
// Macro to get the size of a container, casting it to an int.
#define sz(a) int((a).size())
// Macro for pushing an element to the back of a vector.
#define pb push_back
// Macro for creating a pair.
#define mp make_pair
// Macro to get iterators to the beginning and end of a container for algorithms.
#define all(c) (c).begin(), (c).end()
// Macro for a type-agnostic for-each loop, often used for iterators.
#define tr(c, i) for(typeof((c).begin() i = (c).begin(); i != (c).end(); i++)
// Macro to check if an element is present in an associative container (e.g., map, set).
#define present(c, x) ((c).find(x) != (c).end())
// Macro to check if an element is present in a sequence container (e.g., vector, list).
#define cpresent(c, x) (find(all(c), x) != (c).end())

using namespace std;

/**
 * @brief Solves the domino piling problem by calculating the maximum number of dominoes.
 * This function reads the dimensions of the chessboard (M and N) and computes the maximum
 * number of 2x1 dominoes that can be placed on it.
 */
void solve() {
  int m, n, ans;
  cin >> m >> n;
  /**
   * @brief Calculates the maximum number of dominoes based on chessboard dimensions.
   * Logic: If either dimension is even, the board can be perfectly tiled, and the answer is (M*N)/2.
   * If both dimensions are odd, one square must be left uncovered. The strategy is to remove
   * one row or one column, tile the remaining even-sized area, and then try to tile the
   * removed row/column as much as possible, taking the maximum of two configurations.
   */
  if (n % 2 == 0 || m % 2 == 0) {
    ans = (m * n) / 2;
  } else {
    // For odd x odd boards, the formula simplifies to (m*n - 1) / 2 as one square will always remain.
    // The given formula is an expanded version that calculates this implicitly.
    ans = max(m * (n - 1) / 2 + m / 2, n * (m - 1) / 2 + n / 2);
  }
  cout << ans << "\n";
}

int main() {
  /**
   * Functional Utility: Decouples C++ streams from standard C streams and unties cin from cout.
   * This optimizes high-throughput I/O operations, which is common in competitive programming
   * to meet strict time limits.
   */
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  solve();
  return 0;
}