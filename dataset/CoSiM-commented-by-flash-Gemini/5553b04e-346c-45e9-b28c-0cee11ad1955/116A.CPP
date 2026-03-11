/**
 * @file 116A.CPP
 * @brief Solution to Codeforces Problem 116A: "Tram".
 *        The problem simulates a tram moving along a route with multiple stops.
 *        At each stop, some passengers exit and some enter. The goal is to
 *        find the maximum number of passengers that were in the tram at any
 *        single moment, which represents the minimum capacity the tram should have.
 *
 * Algorithm: Simulation and tracking maximum. The program iterates through each tram stop.
 *            For each stop, it updates the current number of passengers by subtracting
 *            exiting passengers and adding entering passengers. It continuously tracks
 *            the maximum number of passengers encountered so far.
 * Time Complexity: O(N) where N is the number of tram stops, as the solution involves
 *                  a single pass through the stops.
 * Space Complexity: O(1) as only a few variables are used to store current state and maximum.
 */
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>
using namespace std;
int main() {
  /**
   * Functional Utility: Declaration and initialization of variables.
   * `up`: Number of passengers entering the tram at a stop.
   * `down`: Number of passengers exiting the tram at a stop.
   * `count`: Stores the maximum number of passengers observed in the tram so far.
   * `i`: Loop counter for iterating through stops.
   * `j`: Unused variable, potential for removal.
   * `a`: Total number of tram stops.
   * `sum`: Current number of passengers in the tram.
   */
  int up, down, count = 0;
  int i, j, a, sum = 0;
  cin >> a; // Input: Read the total number of tram stops.

  /**
   * Block Logic: Iterates through each tram stop to simulate passenger changes.
   * Pre-condition: `a` is the total number of stops. `sum` is 0, `count` is 0.
   * Invariant: `sum` holds the current number of passengers in the tram; `count`
   *            holds the maximum passengers observed up to the current stop.
   */
  for (i = 1; i <= a; i++) {
    // Input: Read the number of passengers exiting (`down`) and entering (`up`) at the current stop.
    cin >> down >> up;
    // Functional Utility: Calculate the net change in passengers at the current stop.
    // Update `up` to represent the actual number of passengers remaining after exits.
    up = up - down;
    // Functional Utility: Update the total number of passengers currently in the tram.
    sum = sum + up;
    /**
     * Conditional Logic: Checks if the current number of passengers (`sum`)
     * exceeds the previously recorded maximum (`count`).
     * If true, update `count` to the new maximum.
     */
    if (sum > count) {
      count = sum;
    }
  }
  // Output: Print the maximum number of passengers observed in the tram, followed by a newline.
  cout << count << endl;

  return 0;
}
