/**
 * @file 116A.CPP
 * @brief Solution to a problem calculating the maximum passenger capacity
 *        needed for a tram. Given the number of stops and for each stop,
 *        the number of passengers exiting and entering, the program
 *        determines the maximum number of passengers simultaneously present
 *        in the tram.
 *
 * Algorithm: Simulation with running total.
 *            The algorithm simulates the tram's journey, maintaining a
 *            running total of passengers inside. At each stop, it updates
 *            this total based on exiting and entering passengers and
 *            tracks the maximum value reached by the running total.
 * Time Complexity: O(N), where N is the number of tram stops.
 *                  The loop iterates N times, performing constant-time
 *                  arithmetic operations in each iteration.
 * Space Complexity: O(1), as only a few variables are used to store counts.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  // Functional Utility: Initializes `n` for number of stops, `a` for exiting passengers,
  // `b` for entering passengers, `max` for maximum capacity, and `tot` for current total passengers.
  int n, a, b, max = 0, tot = 0;
  // Block Logic: Reads the number of tram stops from standard input.
  cin >> n;

  /**
   * Block Logic: Iterates through each tram stop, processing passenger changes.
   * Invariant: `n` decreases with each iteration, `tot` holds the current number
   *            of passengers, and `max` holds the peak passenger count observed so far.
   */
  while (n--) {
    // Block Logic: Reads the number of passengers exiting (`a`) and entering (`b`)
    // at the current stop.
    cin >> a >> b;
    // Functional Utility: Updates the total number of passengers by subtracting those exiting
    // and adding those entering.
    tot += -a + b;
    // Pre-condition: Checks if the current total passenger count exceeds the recorded maximum.
    if (max < tot)
      // Block Logic: Updates `max` if `tot` is a new peak.
      max = tot;
  }
  // Post-condition: Prints the overall maximum number of passengers observed in the tram.
  cout << max << endl;
  return 0;
}
