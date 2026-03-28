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
using namespace std;
int main() {
  // Declare variables: 'i' for loop counter, 'max' for maximum passengers observed (initialized to 0),
  // 'a' for passengers exiting, 'b' for passengers entering,
  // 'curr' for current passenger count (initialized to 0), and 'n' for number of stops.
  int i, max(0), a, b, curr(0), n;
  // Read the total number of stops 'n'.
  cin >> n;
  /**
   * Block Logic: Loop through each tram stop to simulate passenger changes.
   * Invariant: 'i' increments, processing each stop; 'curr' reflects the current
   *            passenger count, while 'max' stores the highest 'curr' seen so far.
   */
  for (i = 0; i < n; i++) {
    // Read the number of passengers exiting 'a' and entering 'b' at the current stop.
    cin >> a >> b;
    // Update the current passenger count by subtracting exits and adding entries.
    curr += -a + b;
    /**
     * Block Logic: Checks if the current passenger count 'curr' exceeds the previously
     *            recorded maximum 'max', and updates 'max' if necessary.
     * Pre-condition: 'curr' has been updated with the passenger changes for the current stop.
     */
    if (curr > max)
      // If current count is greater than the maximum, update the maximum.
      max = curr;
  }
  // Print the overall maximum number of passengers observed in the tram.
  cout << max << endl;
  return 0; // Indicate successful program execution.
}
