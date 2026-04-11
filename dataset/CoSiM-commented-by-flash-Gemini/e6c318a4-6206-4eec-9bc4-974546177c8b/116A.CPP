/**
 * @file 116A.CPP
 * @brief This program calculates the maximum capacity needed for a tram.
 * It simulates the entry and exit of passengers at various stops.
 * The current number of passengers in the tram is tracked, and the maximum
 * value achieved during the journey is the required capacity.
 *
 * Algorithm: Iterative simulation of tram stops to find peak occupancy.
 * Time Complexity: O(N) where N is the number of tram stops.
 * Space Complexity: O(1)
 */

#include <iostream>
using namespace std;

int main() {
  // Functional Utility: 'stn' (stations/stops) stores the total number of tram stops.
  int stn;
  // Block Logic: Reads the total number of tram stops 'stn' from standard input.
  cin >> stn;

  // Functional Utility: 'a' stores the number of passengers exiting at a stop.
  int a;
  // Functional Utility: 'b' stores the number of passengers entering at a stop.
  int b;
  // Functional Utility: 'sum' tracks the current number of passengers in the tram.
  int sum = 0;
  // Functional Utility: 'max' stores the maximum number of passengers observed in the tram.
  int max = 0;

  /**
   * Block Logic: Iterates through each tram stop to simulate passenger changes.
   * Invariant: At the start of each iteration, 'stn' represents the remaining stops to process.
   */
  while (stn--) {
    // Block Logic: Reads the number of passengers exiting ('a') and entering ('b') at the current stop.
    cin >> a >> b;
    // Functional Utility: Updates the current number of passengers in the tram.
    // Inline: Passengers exiting are subtracted, and passengers entering are added.
    sum += (b - a);
    /**
     * Block Logic: Checks if the current number of passengers exceeds the previously recorded maximum.
     * Pre-condition: 'sum' reflects the passenger count after the current stop.
     * Invariant: 'max' always holds the highest passenger count encountered so far.
     */
    if (max < sum) {
      max = sum;
    }
  }
  // Output the maximum capacity required for the tram.
  cout << max;
  return 0;
}
