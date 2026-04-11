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

#include <bits/stdc++.h>

using namespace std;

int main() {
  // Functional Utility: Declares an integer 'n' to store the number of tram stops.
  int n;
  // Functional Utility: 'ai' stores the number of passengers exiting at a stop.
  int ai;
  // Functional Utility: 'bi' stores the number of passengers entering at a stop.
  int bi;
  // Functional Utility: 'inTram' tracks the current number of passengers in the tram.
  int inTram = 0;
  // Functional Utility: 'max' stores the maximum number of passengers observed in the tram.
  int max = 0;

  // Block Logic: Reads the total number of tram stops 'n' from standard input.
  cin >> n;

  /**
   * Block Logic: Iterates through each tram stop to simulate passenger changes.
   * Invariant: At the start of each iteration, 'n' represents the remaining stops to process.
   */
  while (n--) {
    // Block Logic: Reads the number of passengers exiting ('ai') and entering ('bi') at the current stop.
    cin >> ai >> bi;
    // Functional Utility: Updates the current number of passengers in the tram.
    // Inline: Passengers exiting are subtracted, and passengers entering are added.
    inTram += (-ai + bi);
    /**
     * Block Logic: Checks if the current number of passengers exceeds the previously recorded maximum.
     * Pre-condition: 'inTram' reflects the passenger count after the current stop.
     * Invariant: 'max' always holds the highest passenger count encountered so far.
     */
    if (inTram > max)
      max = inTram;
  }

  // Output the maximum capacity required for the tram.
  cout << max << endl;

  return 0;
}
