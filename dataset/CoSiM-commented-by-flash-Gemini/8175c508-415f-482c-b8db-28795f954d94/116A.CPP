/**
 * @file 116A.CPP
 * @brief Solution to Codeforces problem 116A - Tram.
 *        This program simulates the movement of passengers in a tram and calculates
 *        the maximum number of passengers that were simultaneously inside the tram.
 *
 * Algorithm:
 *   1. Reads `n`, the total number of tram stops.
 *   2. Initializes `c` to 0 (representing the number of passengers currently in the tram)
 *      and `max` to 0 (to store the highest `c` observed).
 *   3. Iterates for each tram stop:
 *      a. Reads the number of passengers exiting (`l`) and entering (`e`).
 *      b. Updates `c` by subtracting `l` and adding `e`.
 *      c. Compares `c` with `max` and updates `max` if `c` is higher.
 *   4. After processing all stops, prints the final `max`.
 *
 * Time Complexity: O(n), where `n` is the number of tram stops, due to the single loop.
 * Space Complexity: O(1) for storing a few integer variables.
 */

#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  /**
   * @brief Main function that solves the "Tram" problem.
   *
   * Reads the number of stops and passenger changes at each stop to determine
   * the maximum passenger count observed in the tram.
   *
   * @param argc Argument count (unused).
   * @param argv Argument vector (unused).
   * @return int Returns 0 upon successful execution.
   */
  int n; // Total number of tram stops.
  cin >> n; // Reads the number of stops.

  int c = 0;   // `c` represents the current number of passengers in the tram.
  int max = 0; // `max` stores the maximum number of passengers observed.
  int l, e;    // `l` (passengers leaving), `e` (passengers entering).
  /**
   * Block Logic: Iterates through each tram stop to process passenger changes.
   * Invariant: `c` reflects the current passenger count. `max` holds the peak capacity.
   * Pre-condition: `n` is a non-negative integer.
   */
  while (n--) { // Loop continues for `n` iterations, decrementing `n` each time.
    cin >> l >> e; // Reads passenger changes for the current stop.
    c -= l;        // Passengers exit.
    c += e;        // Passengers enter.
    /**
     * Conditional Logic: Updates `max` if the `c` (current capacity) is higher.
     */
    if (c > max)
      max = c; // Records new peak capacity.
  }
  // Prints the highest observed passenger capacity.
  cout << max;
  return 0;
}
