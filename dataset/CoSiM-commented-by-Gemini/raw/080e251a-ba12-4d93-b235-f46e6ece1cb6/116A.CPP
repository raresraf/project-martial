/**
 * @file 116A.CPP
 * @brief Calculates the minimum required capacity of a tram.
 * @details This program solves the "Tram" problem from Codeforces. It simulates the
 * number of passengers on a tram at each stop to find the maximum number of
 * passengers present at any one time. This maximum value represents the minimum
 * capacity the tram must have.
 *
 * @see https://codeforces.com/problemset/problem/116/A
 *
 * Algorithm:
 * 1. Read the total number of tram stops.
 * 2. Initialize a 'total' passenger count and a 'max' capacity tracker to zero.
 * 3. Loop for each stop:
 *    a. Read the number of passengers exiting and entering.
 *    b. Update the 'total' passenger count by adding the entering and subtracting the exiting passengers.
 *    c. If the new 'total' is greater than the current 'max', update 'max' to this new total.
 * 4. After iterating through all stops, 'max' will hold the minimum required capacity. Print 'max'.
 *
 * Time Complexity: O(N), where N is the number of stops, due to a single loop through all stops.
 * Space Complexity: O(1), as the program uses a fixed number of variables.
 */

#include <cstdio>

int main() {
  // The total number of stops on the tram line.
  int numStops = 0;
  scanf("%d", &numStops);

  // total: current number of passengers on the tram.
  // max: the maximum number of passengers observed so far (the required capacity).
  // in: passengers entering at the current stop.
  // out: passengers exiting at the current stop.
  int total = 0, max = 0, in = 0, out = 0;

  /**
   * @brief Main loop to simulate passenger flow at each stop.
   *
   * Pre-condition: numStops contains the number of stops to simulate.
   * Invariant: After each iteration, 'total' is the current number of passengers,
   *            and 'max' is the maximum number of passengers seen up to that point.
   */
  while (numStops--) {
    scanf("%d %d", &out, &in);
    // Update the current number of passengers on the tram.
    total += in - out;

    // If the current passenger count exceeds the max seen so far, update the max.
    if (total > max)
      max = total;
  }

  // Print the minimum required capacity.
  printf("%d", max);
  return 0;
}
