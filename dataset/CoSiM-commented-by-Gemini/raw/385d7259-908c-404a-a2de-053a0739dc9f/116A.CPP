/**
 * @file 116A.CPP
 * @brief Calculates the minimum possible capacity of a tram.
 *
 * This program simulates the passenger count on a tram over a series of stops.
 * At each stop, a number of passengers exit and a number of passengers enter.
 * The goal is to find the maximum number of passengers on the tram at any given
 * point in time, which corresponds to the minimum capacity the tram must have.
 *
 * Algorithm:
 * 1. Read an integer `n`, the number of tram stops.
 * 2. Initialize `in` (current passengers inside) and `temp` (maximum passengers seen so far) to 0.
 * 3. Loop `n` times, once for each stop.
 *    a. Read the number of passengers exiting (`a`) and entering (`b`).
 *    b. Update the current number of passengers: `in = in - a + b`.
 *    c. Compare the new `in` value with `temp`. If `in` is greater, update `temp` to `in`.
 * 4. After the loop, `temp` will hold the maximum passenger count, which is the minimum required capacity.
 * 5. Print the value of `temp`.
 *
 * Time Complexity: O(N), where N is the number of stops, due to a single loop through the stops.
 * Space Complexity: O(1), as the program only uses a few variables to store the running state.
 */
#include <stdio.h>
#include <string.h>

int main() {
  int i, n, a, b, t1 = 0, temp = 0, in = 0;
  // Read the total number of stops.
  scanf("%d", &n);

  /**
   * @brief Main simulation loop that processes each stop.
   * Pre-condition: `n` is the number of stops to simulate.
   * Invariant: At the start of each iteration, `in` is the number of passengers
   * on the tram, and `temp` is the maximum number seen so far.
   */
  for (i = 0; i < n; i++) {
    // Read passengers exiting (a) and entering (b).
    scanf("%d %d", &a, &b);
    // Update the number of passengers currently inside.
    in = b - a + in;
    /**
     * @brief Update the required capacity.
     * If the current number of passengers is the highest yet,
     * it becomes the new minimum required capacity.
     */
    if (i == 0 || in > temp) {
      temp = in;
    }
  }
  // Print the final calculated minimum capacity.
  printf("%d", temp);

  return 0;
}