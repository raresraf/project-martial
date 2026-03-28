/**
 * @file 116A.CPP
 * @brief Solution to a problem that calculates the maximum number of passengers
 *        ever present in a tram/bus during its journey.
 *
 * This program simulates a tram's journey through several stops. At each stop,
 * a certain number of passengers exit (a) and another number enter (b). The goal is
 * to determine the maximum number of passengers present in the tram at any single
 * moment throughout its route. This is a common problem used to test basic
 * simulation and state tracking logic.
 *
 * Algorithm: Iterative simulation of the tram journey, maintaining a running count
 *            of current passengers and updating a peak capacity observed.
 * Time Complexity: O(N), where N is the number of stops, as each stop is processed once.
 * Space Complexity: O(1) for storing variables.
 */
#include <iostream>
#include <algorithm> // Required for std::max
using namespace std;

int main() {
  // Declare 'n' for the number of stops.
  // 'currentPass' tracks the number of passengers currently in the tram, initialized to 0.
  // 'maxCap' stores the maximum number of passengers observed, initialized to 0.
  int n, currentPass = 0, maxCap = 0;
  // Read the total number of stops 'n'.
  cin >> n;

  /**
   * Block Logic: Loop through each tram stop to simulate passenger changes.
   * Invariant: 'n' decreases with each iteration; 'currentPass' reflects the current
   *            passenger count, while 'maxCap' stores the highest 'currentPass' seen so far.
   */
  while (n--) {
    // Declare 'a' for passengers exiting and 'b' for passengers entering at the current stop.
    int a, b;
    // Read the number of passengers exiting 'a' and entering 'b'.
    cin >> a >> b;

    // Update the current passenger count: subtract exits and add entries.
    currentPass += b - a;
    // Update 'maxCap' to be the greater of its current value and 'currentPass'.
    // This ensures 'maxCap' always holds the peak number of passengers.
    maxCap = max(maxCap, currentPass);
  }

  // Output the overall maximum number of passengers observed in the tram.
  cout << maxCap << endl;
  return 0; // Indicate successful program execution.
}
