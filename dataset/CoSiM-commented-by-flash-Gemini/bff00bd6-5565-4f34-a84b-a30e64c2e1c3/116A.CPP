/**
 * @file 116A.CPP
 * @brief Solution to a problem that calculates the maximum number of passengers
 *        ever present in a tram/bus during its journey.
 *
 * This program simulates a tram's journey through several stops. At each stop,
 * a certain number of passengers exit, and another number enter. The goal is
 * to determine the maximum number of passengers present in the tram at any single
 * moment. This is a common problem used to test basic simulation and state tracking.
 *
 * Algorithm: Iterative simulation, tracking the current passenger count and
 *            updating a maximum observed count.
 * Time Complexity: O(N), where N is the number of stops.
 * Space Complexity: O(1) for variable storage.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // Declare variables: 'i' for loop counter, 'n' for number of stops,
  // 'a' for passengers exiting, 'b' for passengers entering.
  // 'cnt' for current passenger count (initialized to 0),
  // 'sum' for maximum passengers observed (initialized to 0).
  int i, n, a, b, cnt = 0, sum = 0;
  // Read the total number of stops 'n'.
  cin >> n;
  /**
   * Block Logic: Loop through each tram stop to simulate passenger changes.
   * Invariant: 'i' increments, processing each stop; 'cnt' reflects the current
   *            passenger count, while 'sum' stores the highest 'cnt' seen so far.
   */
  for (i = 0; i < n; i++) {
    // Read the number of passengers exiting 'a' and entering 'b' at the current stop.
    cin >> a >> b;
    // Update the current passenger count: add entries and then subtract exits.
    cnt = (cnt + b) - a;
    /**
     * Block Logic: Checks if the current passenger count 'cnt' exceeds or equals
     *            the previously recorded maximum 'sum', and updates 'sum' if necessary.
     * Pre-condition: 'cnt' has been updated with the passenger changes for the current stop.
     */
    if (cnt >= sum) {
      // If current count is greater than or equal to the maximum, update the maximum.
      sum = cnt;
    }
  }
  // Output the overall maximum number of passengers observed in the tram.
  cout << sum;
  return 0;
}
