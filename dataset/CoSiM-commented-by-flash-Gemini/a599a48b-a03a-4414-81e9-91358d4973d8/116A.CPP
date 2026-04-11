/**
 * @file 116A.CPP
 * @brief This program calculates the maximum capacity needed for a tram,
 * simulating passenger entry and exit at various stops. It tracks the current
 * number of passengers and identifies the peak occupancy.
 *
 * Algorithm: Iterative simulation of tram stops to find peak occupancy.
 * Time Complexity: O(X) where X is the number of tram stops.
 * Space Complexity: O(1)
 */

#include <iostream>
using namespace std;

int main() {
  // Functional Utility: 'x' initially stores the total number of tram stops,
  //                     and decrements in the loop to control iterations.
  int x;
  // Functional Utility: 'n1' stores the number of passengers exiting at a stop.
  int n1;
  // Functional Utility: 'n2' stores the number of passengers entering at a stop.
  int n2;
  // Functional Utility: 'temp1' tracks the current number of passengers in the tram.
  int temp1 = 0;
  // Functional Utility: 'temp2' stores the maximum number of passengers observed in the tram.
  int temp2 = 0;

  // Block Logic: Reads the total number of tram stops 'x' from standard input.
  cin >> x;

  /**
   * Block Logic: Iterates through each tram stop to simulate passenger changes.
   * Invariant: At the start of each iteration, 'x' represents the remaining stops to process.
   */
  while (x > 0) {
    // Block Logic: Reads the number of passengers exiting ('n1') and entering ('n2') at the current stop.
    cin >> n1 >> n2;
    // Functional Utility: Updates the current number of passengers in the tram.
    // Inline: Passengers exiting are subtracted, and passengers entering are added.
    temp1 = n2 - n1 + temp1;
    /**
     * Block Logic: Checks if the current number of passengers exceeds the previously recorded maximum.
     * Pre-condition: 'temp1' reflects the passenger count after the current stop.
     * Invariant: 'temp2' always holds the highest passenger count encountered so far.
     */
    if (temp1 > temp2) {
      temp2 = temp1;
    }
    // Functional Utility: Decrements the counter for remaining stops.
    x--;
  }
  // Output the maximum capacity required for the tram.
  cout << temp2 << endl;
  return 0;
}
