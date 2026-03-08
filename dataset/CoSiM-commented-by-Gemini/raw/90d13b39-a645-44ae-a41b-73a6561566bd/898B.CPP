/**
 * @file 898B.CPP
 * @brief Solution for a linear Diophantine equation problem (e.g., Codeforces 898B "Proper Nutrition").
 * @details The problem asks to find non-negative integers x and y that satisfy the
 * equation `a*x + b*y = n` for given a, b, and n. This program uses a brute-force
 * approach to find a valid pair (x, y).
 *
 * @algorithm Brute-force search for a Diophantine equation solution.
 * @complexity
 * - Time: O(n/a). The loop iterates through possible values of 'x' (represented by 'i').
 *   The maximum useful value for 'i' is `n/a`, so the loop runs approximately `n/a` times.
 * - Space: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<int, int> ii; // This pair typedef is unused.

int main() {
  // Fast I/O setup, common in competitive programming.
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);

  ll n, a, b;
  cin >> n >> a >> b;

  /**
   * @block Solution Search
   * @logic Iterates through all possible non-negative values for the number of 'a's ('i').
   * For each 'i', it calculates the required number of 'b's ('j') and checks if it's a
   * valid non-negative integer solution.
   */
  for (int i = 0; i <= n; i++) {
    // If we use `i` items of cost `a`, the remaining value is `n - a * i`.
    // Check if this remainder is non-negative and perfectly divisible by `b`.
    if ((n - a * i) >= 0 && (n - a * i) % b == 0) {
      ll j = (n - a * i) / b;
      // We have found a valid pair of non-negative integers (i, j).
      cout << "YES" << endl;
      cout << i << " " << j << endl;
      return 0; // Exit after finding the first solution.
    }
  }

  // If the loop completes, no solution was found.
  cout << "NO" << endl;
  return 0;
}
