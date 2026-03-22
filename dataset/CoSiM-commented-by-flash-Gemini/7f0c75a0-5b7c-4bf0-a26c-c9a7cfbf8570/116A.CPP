/**
 * @file 116A.CPP
 * @brief Solution to Codeforces problem 116A - Tram.
 *        This program simulates the movement of passengers in a tram and calculates
 *        the maximum number of passengers that were simultaneously inside the tram.
 *
 * Algorithm:
 *   1. Reads `amt`, the total number of tram stops.
 *   2. Initializes `now` to 0 (representing the number of passengers currently in the tram)
 *      and `maxn` to -1 (to store the highest `now` observed).
 *   3. Iterates for each tram stop:
 *      a. Reads the number of passengers exiting (`out`) and entering (`in`).
 *      b. Updates `now` by subtracting `out` and adding `in`.
 *      c. Compares `now` with `maxn` and updates `maxn` if `now` is higher.
 *   4. After processing all stops, prints the final `maxn`.
 *
 * Time Complexity: O(amt), where `amt` is the number of tram stops, due to the single loop.
 * Space Complexity: O(1) for storing a few integer variables.
 */

#include <bits/stdc++.h> // Includes common C++ standard library headers.
using namespace std;

int main() {
  /**
   * @brief Main function that solves the "Tram" problem.
   *
   * Reads the number of stops and passenger changes at each stop to determine
   * the maximum passenger count observed in the tram.
   *
   * @return int Returns 0 upon successful execution.
   */
  int amt; // Total number of tram stops.
  int maxn = -1; // Stores the maximum number of passengers seen. Initialized to -1.
                 // A safer initialization would be 0 as passenger count cannot be negative.
  int i;         // Loop counter.
  int out, in;   // Passengers exiting and entering at a stop.
  int now = 0;   // Represents the current number of passengers in the tram.
  cin >> amt;    // Reads the total number of stops.
  /**
   * Block Logic: Iterates through each tram stop to process passenger changes.
   * Invariant: `now` reflects the current passenger count. `maxn` holds the peak count.
   * Pre-condition: `amt` is a non-negative integer.
   */
  for (i = 1; i <= amt; i++) {
    cin >> out >> in;     // Reads passenger changes for the current stop.
    now += in - out;      // Updates the current number of passengers.
    /**
     * Conditional Logic: Updates `maxn` if the `now` (current capacity) is higher.
     */
    maxn = max(maxn, now); // Compares current passengers with maximum observed and updates.
  }
  // Prints the highest observed passenger capacity.
  cout << maxn << endl;
  return 0;
}
