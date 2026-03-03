/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A - "Tram".
 * This program calculates the minimum capacity a tram must have to carry all passengers
 * through a series of stops, where passengers exit and enter at each stop.
 * It tracks the current number of passengers and finds the peak occupancy.
 * Algorithm: Iterative simulation, tracking current and maximum passenger counts.
 * Time Complexity: O(N) - where N is the number of tram stops.
 * Space Complexity: O(1) - uses a constant amount of memory for variables.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries (e.g., iostream).
using namespace std;     // Uses the standard namespace.

int main() {
  // n: total number of tram stops.
  // a: passengers exiting at current stop.
  // b: passengers entering at current stop.
  // c: current number of passengers in the tram.
  // m: maximum number of passengers observed in the tram (required capacity).
  int n, a, b, c(0), m(0);

  // Reads the total number of tram stops.
  cin >> n;

  /**
   * Block Logic: Simulates passenger movements at each tram stop.
   * This loop processes each of the 'n' stops, updating the current passenger count
   * and tracking the maximum occupancy reached.
   * Invariant: `c` always holds the current number of passengers, and `m` holds
   *            the maximum `c` has been so far.
   */
  while (n--) { // Loop 'n' times, decrementing 'n' in each iteration.
    cin >> a >> b; // Reads passengers exiting ('a') and entering ('b').
    c = c - a;     // Subtracts exiting passengers.
    c = c + b;     // Adds entering passengers.

    // Updates 'm' if the current passenger count 'c' is higher than the previously recorded maximum.
    if (c > m) {
      m = c;
    }
  }

  // Prints the maximum capacity required for the tram.
  cout << m;

  return 0; // Indicates successful program execution.
}