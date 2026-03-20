/**
 * @file 116A.CPP
 * @brief An efficient solution for Codeforces problem 116A, "Tram".
 * @details This program finds the minimum required capacity of a tram by processing
 * passenger data in a memory-efficient, on-the-fly manner.
 *
 * @section Algorithm
 * The solution simulates the changing number of passengers in the tram to find the
 * peak occupancy, which corresponds to the minimum required capacity. This is done
 * without storing the data for each stop.
 *
 * 1. Read the number of stops, `n`.
 * 2. Initialize `total` (current passengers) to 0 and `mx` (maximum capacity)
 *    to a very small number.
 * 3. Loop `n` times:
 *    a. In each iteration, read the number of passengers exiting and entering.
 *    b. Update the `total` by adding the entering passengers and subtracting
 *       the exiting ones.
 *    c. Compare the new `total` with `mx` and update `mx` if `total` is larger.
 * 4. After the loop, `mx` holds the maximum number of passengers that were in
 *    the tram at any one time, which is the required capacity.
 *
 * @subsection Complexity
 * - Time Complexity: O(n), where `n` is the number of stops, because of the
 *   single loop.
 * - Space Complexity: O(1), as it processes the data for each stop without
 *   storing it, using only a few variables.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // n: number of tram stops.
  // enter, exit: number of passengers entering and exiting at a stop.
  int n, enter, exit;
  // total: tracks the current number of passengers inside the tram.
  int total = 0;
  // mx: tracks the maximum number of passengers observed, initialized to a very low value.
  int mx = -999999;
  cin >> n;
  /**
   * @brief This loop processes each stop's data on-the-fly.
   * Pre-condition: `total` is 0, representing an empty tram at the start.
   * Invariant: At the end of each iteration, `total` is the current number of
   * passengers, and `mx` is the maximum number seen up to that point.
   */
  for (int i = 0; i < n; i++) {
    cin >> exit >> enter;
    // Update the current passenger count.
    total = total + enter;
    total = total - exit;
    // If the current passenger count exceeds the max seen so far, update the max.
    if (total > mx) {
      mx = total;
    }
  }
  // Print the calculated minimum capacity.
  cout << mx;
}
