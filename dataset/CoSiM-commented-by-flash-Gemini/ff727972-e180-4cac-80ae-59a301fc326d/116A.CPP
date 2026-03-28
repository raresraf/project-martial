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
#include <iostream> // Required for std::cout (though printf is used for final output).
#include <algorithm> // Required for std::max.
using namespace std; // Using the standard namespace.

int main() {
  // Declare 'n' for the number of stops, 'a' for passengers exiting, 'b' for passengers entering.
  // 'ma' (max) stores the maximum number of passengers observed, initialized to 0.
  // 'cur' (current) tracks the number of passengers currently in the tram, initialized to 0.
  int n, a, b, ma = 0, cur = 0;
  // Read the total number of stops 'n'.
  scanf("%d", &n);
  /**
   * Block Logic: Loop through each tram stop to simulate passenger changes.
   * Invariant: 'i' increments through the stops; 'cur' reflects the current
   *            passenger count, while 'ma' stores the highest 'cur' seen so far.
   */
  for (register int i = 0; i < n; i++) { // 'register' is a hint for compiler optimization, often ignored by modern compilers.
    // Read the number of passengers exiting 'a' and entering 'b' at the current stop.
    scanf("%d", &a);
    scanf("%d", &b);
    // Update the current passenger count: first subtract exits.
    cur = cur - a;
    // Then add entries.
    cur = cur + b;
    // Update 'ma' to be the greater of its current value and 'cur'.
    // This ensures 'ma' always holds the peak number of passengers.
    ma = max(cur, ma);
  }
  // Print the overall maximum number of passengers observed in the tram.
  printf("%d", ma);
  return 0; // Indicate successful program execution (implicitly if control reaches end of main).
}