/**
 * @file 116A.CPP
 * @brief An optimal O(1) space solution for the "Tram" capacity problem.
 *
 * @details
 * This program calculates the minimum required capacity of a tram by simulating
 * passenger flow in a single pass. It iteratively updates the current number of
 * passengers and tracks the maximum number seen so far. This approach is highly
 * efficient as it does not require storing data for all stops.
 *
 * @section
 * Algorithm: Single-pass simulation with real-time maximum tracking.
 * Time Complexity: O(N), where N is the number of stops.
 * Space Complexity: O(1).
 */
#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  // n: the total number of stops.
  // t: a temporary variable to read exit and enter counts.
  // m: tracks the maximum number of passengers seen so far (the required capacity).
  // s: the current sum of passengers on the tram.
  int n, t, m = 0, s = 0;
  cin >> n;

  /**
   * @brief A single loop to simulate all stops and find the peak passenger count.
   *
   * Logic: This loop processes one stop at a time in an O(1) space fashion.
   * It updates the current passenger count ('s') and immediately checks if this
   * new count is a peak ('m').
   */
  while (n--) {
    // Read number of exiting passengers into 't' and update the sum.
    cin >> t;
    s -= t;
    // Read number of entering passengers into the same 't' and update the sum.
    cin >> t;
    s += t;
    // Update the maximum capacity if the current count is a new high.
    m = max(m, s);
  }
  // 'm' now holds the highest passenger count recorded, which is the answer.
  cout << m;
  return 0;
}
