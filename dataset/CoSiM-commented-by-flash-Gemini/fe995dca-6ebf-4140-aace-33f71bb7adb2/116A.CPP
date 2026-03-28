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
#include <stdio.h> // Required for scanf and printf.
using namespace std; // Using the standard namespace.

int main() {
  // Declare variables: 'n' for the number of stops, 'a' for passengers exiting,
  // 'b' for passengers entering.
  // 'temp' for current passenger count (initialized to 0).
  // 'max' for maximum passengers observed (initialized to 0).
  int n, a, b, temp = 0, max = 0;
  // Read the total number of stops 'n'.
  scanf("%d", &n);
  /**
   * Block Logic: Loop through each tram stop to simulate passenger changes.
   * Invariant: 'n' decreases with each iteration; 'temp' reflects the current
   *            passenger count, while 'max' stores the highest 'temp' seen so far.
   */
  while (n--) {
    // Read the number of passengers exiting 'a' and entering 'b' at the current stop.
    scanf("%d %d", &a, &b);
    // Update the current passenger count by subtracting exits and adding entries.
    temp += b - a;
    /**
     * Block Logic: Checks if the current passenger count 'temp' exceeds the previously
     *            recorded maximum 'max', and updates 'max' if necessary.
     * Pre-condition: 'temp' has been updated with the passenger changes for the current stop.
     */
    if (temp > max)
      // If current count is greater than the maximum, update the maximum.
      max = temp;
  }
  // Print the overall maximum number of passengers observed in the tram.
  printf("%d", max);
  return 0; // Indicate successful program execution.
}
