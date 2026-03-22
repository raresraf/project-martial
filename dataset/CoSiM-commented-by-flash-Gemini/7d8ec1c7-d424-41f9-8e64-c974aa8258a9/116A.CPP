/**
 * @file 116A.CPP
 * @brief Solution to Codeforces problem 116A - Tram.
 *        This program simulates the movement of passengers in a tram and calculates
 *        the maximum number of passengers that were simultaneously inside the tram.
 *
 * Algorithm:
 *   1. Reads `total_stops`, the total number of tram stops.
 *   2. Initializes `current_capacity` to 0 (representing the number of passengers
 *      currently in the tram) and `max_capacity` to 0 (to store the highest
 *      `current_capacity` observed).
 *   3. Iterates for each tram stop:
 *      a. Reads the number of passengers exiting (`exits`) and entering (`entries`).
 *      b. Updates `current_capacity` by subtracting `exits` and adding `entries`.
 *      c. Compares `current_capacity` with `max_capacity` and updates `max_capacity`
 *         if `current_capacity` is higher.
 *   4. After processing all stops, prints the final `max_capacity`.
 *
 * Time Complexity: O(total_stops), where `total_stops` is the number of tram stops, due to the single loop.
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
  long long total_stops;    // Total number of tram stops.
  long long current_capacity = 0; // Number of passengers currently in the tram.
  long long max_capacity = 0;     // Maximum number of passengers observed.
  long long entries = 0;          // Passengers entering at a stop.
  long long exits = 0;            // Passengers exiting at a stop.
  cin >> total_stops; // Reads the total number of stops.

  /**
   * Block Logic: Iterates through each tram stop.
   * Invariant: `current_capacity` reflects the number of passengers after processing
   *            the changes at the current stop. `max_capacity` holds the peak capacity.
   * Pre-condition: `total_stops` is a non-negative integer.
   */
  while (total_stops--) {
    cin >> exits >> entries; // Reads passenger changes for the current stop.
    current_capacity -= exits;  // Passengers exit.
    current_capacity += entries; // Passengers enter.
    /**
     * Conditional Logic: Updates `max_capacity` if the `current_capacity` is higher.
     */
    if (current_capacity > max_capacity) {
      max_capacity = current_capacity; // Records new peak capacity.
    }
  }
  // Prints the highest observed passenger capacity.
  cout << max_capacity;
  return 0;
}