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
 * Space Complexity: O(N) for storing `a` and `b` arrays, although could be O(1)
 *                   if inputs are processed without storing them in arrays.
 */

#include <iostream>

using namespace std;

int main() {
  // Functional Utility: Initializes `n` for number of stops, `a` and `b` arrays
  // for exiting and entering passengers (though not strictly necessary as they can be processed on the fly),
  // `i` for loop counter, `sum` for current total passengers, and `max` for maximum capacity.
  int n, a[20000], b[20000], i, sum = 0, max = 0;
  // Block Logic: Reads the number of tram stops from standard input.
  cin >> n;
  /**
   * Block Logic: Iterates through each tram stop, processing passenger changes.
   * Invariant: `i` is the current stop index, `sum` holds the current number
   *            of passengers, and `max` holds the peak passenger count observed so far.
   */
  for (i = 0; i < n; i++) {
    // Block Logic: Reads the number of passengers exiting (`a[i]`) and entering (`b[i]`)
    // at the current stop.
    cin >> a[i];
    // Functional Utility: Updates the total number of passengers by subtracting those exiting.
    sum -= a[i];
    cin >> b[i];
    // Functional Utility: Updates the total number of passengers by adding those entering.
    sum += b[i];
    // Pre-condition: Checks if the current total passenger count exceeds the recorded maximum.
    if (max < sum) {
      // Block Logic: Updates `max` if `sum` is a new peak.
      max = sum;
    }
  }
  // Post-condition: Prints the overall maximum number of passengers observed in the tram.
  cout << max;
  return 0;
}
