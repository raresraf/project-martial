/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem by calculating the minimum required vehicle capacity.
 * @details This program simulates passenger flow over a series of stops to find the maximum
 * number of passengers inside the tram at any given time, which determines the minimum capacity needed.
 *
 * @algorithm The solution employs a single-pass simulation. It iterates through each stop,
 *            adjusts the current passenger count, and updates a running maximum.
 *
 * @time_complexity O(T), where T is the number of tram stops, due to a single loop over the stops.
 * @space_complexity O(1), as only a constant number of variables are used for state tracking.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  // t: The total number of stops to simulate.
  // a: The number of passengers exiting at a stop.
  // b: The number of passengers entering at a stop.
  // ca: State variable tracking the current number of passengers on the tram.
  // ct: State variable for the maximum recorded passenger load, representing the required capacity.
  int t, a, b, ca = 0, ct = 0;
  cin >> t;

  /**
   * Block Logic: Simulates the tram's journey across all 't' stops.
   * Invariant: At the start of each iteration, `ca` holds the passenger count from the previous stop,
   *            and `ct` holds the maximum passenger count observed so far.
   */
  while (t--) {
    cin >> a >> b;
    // Update current passenger count based on exits and entries.
    ca += b;
    ca -= a;
    
    /**
     * Block Logic: Update the required capacity if the current passenger count represents a new maximum.
     * Pre-condition: `ca` reflects the passenger count after the current stop's activity.
     */
    if (ca >= ct) {
      ct = ca;
    }
  }

  // Output the overall minimum required capacity.
  cout << ct << endl;
  return 0;
}
