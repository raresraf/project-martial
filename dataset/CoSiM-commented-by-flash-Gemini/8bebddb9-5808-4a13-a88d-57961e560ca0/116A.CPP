/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A - "Tram".
 * This program calculates the minimum capacity a tram must have to carry all passengers
 * through a series of stops, where passengers exit and enter at each stop.
 * It tracks the current number of passengers and finds the peak occupancy.
 * Algorithm: Simulation. Iteratively updates current passenger count and tracks the maximum.
 * Time Complexity: O(N) - where N is the number of tram stops. Each stop involves a constant number of operations.
 * Space Complexity: O(1) - uses a constant amount of memory for a few integer variables.
 */
#include <algorithm> // Required for std::max
#include <iostream>  // Required for standard input/output operations (cin, cout)
using namespace std; // Uses the standard namespace to avoid prefixing std::

int main() {
  // n: total number of tram stops.
  // t: temporary variable to read passenger counts (exit and enter).
  // m: stores the maximum number of passengers observed in the tram at any point (maximum capacity required).
  // s: stores the current number of passengers in the tram.
  int n, t, m = 0, s = 0;
  
  // Reads the total number of tram stops.
  cin >> n;
  
  /**
   * Block Logic: Simulates passenger movements at each tram stop.
   * This loop processes each of the 'n' stops, updating the current passenger count
   * and tracking the maximum occupancy reached.
   * Invariant: 's' always holds the current number of passengers in the tram,
   *            'm' always holds the maximum 's' has been so far.
   */
  while (n--) { // Loop 'n' times, decrementing 'n' in each iteration.
    cin >> t;    // Reads the number of passengers exiting at the current stop.
    s -= t;      // Decreases current passenger count by those exiting.
    cin >> t;    // Reads the number of passengers entering at the current stop.
    s += t;      // Increases current passenger count by those entering.
    m = max(m, s); // Updates 'm' to be the higher of its current value or the new current passenger count 's'.
  }
  
  // Prints the maximum capacity required for the tram.
  cout << m;
  return 0; // Indicates successful program execution.
}