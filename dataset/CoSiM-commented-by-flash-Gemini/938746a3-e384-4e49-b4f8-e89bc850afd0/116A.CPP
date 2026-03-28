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
#include <iostream>
#include <stdio.h>
using namespace std;
int main() {
  // Declare variables: 'n' for the number of stops, 'a' for passengers exiting,
  // 'b' for passengers entering, 'c' for current passengers (initialized to 0),
  // and 'm' for maximum passengers observed (initialized to 0).
  int n, a, b, c(0), m(0);
  // Read the total number of stops 'n'.
  scanf("%d", &n);
  /**
   * Block Logic: Loop through each tram stop to simulate passenger changes.
   * Invariant: 'n' decreases with each iteration, and 'c' reflects the current
   *            passenger count, while 'm' stores the highest 'c' seen so far.
   */
  while (n--) {
    // Read the number of passengers exiting 'a' and entering 'b' at the current stop.
    scanf("%d%d", &a, &b);
    // Update the current passenger count: subtract exits.
    c -= a;
    // Update the current passenger count: add entries.
    c += b;
    /**
     * Block Logic: Checks if the current passenger count 'c' exceeds the previously
     *            recorded maximum 'm', and updates 'm' if necessary.
     * Pre-condition: 'c' has been updated with the passenger changes for the current stop.
     */
    if (c > m)
      // If current count is greater than the maximum, update the maximum.
      m = c;
  }
  // Print the overall maximum number of passengers observed in the tram.
  printf("%d\n", m);
  return 0;
}
