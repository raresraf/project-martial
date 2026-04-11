/**
 * @file 116A.CPP
 * @brief This program calculates the maximum capacity needed for a tram.
 * It simulates the entry and exit of passengers at various stops.
 * The current number of passengers in the tram is tracked ('currently'),
 * and the maximum value achieved during the journey ('total') is the required capacity.
 *
 * Algorithm: Iterative simulation of tram stops to find peak occupancy.
 * Time Complexity: O(N) where N is the number of tram stops.
 * Space Complexity: O(1)
 */

#include <iostream>
using namespace std;

int main() {
  // Functional Utility: 'currently' tracks the current number of passengers in the tram.
  int currently = 0;
  // Functional Utility: 'total' stores the maximum number of passengers observed in the tram throughout the journey.
  int total = 0;
  // Functional Utility: 'n' stores the total number of tram stops.
  int n;

  // Block Logic: Reads the total number of tram stops 'n' from standard input.
  cin >> n;

  // Functional Utility: 'a' stores the number of passengers exiting at a stop.
  int a;
  // Functional Utility: 'b' stores the number of passengers entering at a stop.
  int b;

  /**
   * Block Logic: Iterates through each tram stop to simulate passenger changes.
   * Invariant: 'i' represents the current stop number, from 0 to n-1.
   */
  for (int i = 0; i < n; i++) {
    // Block Logic: Reads the number of passengers exiting ('a') and entering ('b') at the current stop.
    cin >> a >> b;
    // Functional Utility: Updates the current number of passengers in the tram.
    // Inline: Passengers exiting are subtracted, and passengers entering are added.
    currently -= a;
    currently += b;
    /**
     * Block Logic: Checks if the current number of passengers ('currently') exceeds the
     * previously recorded maximum ('total').
     * Pre-condition: 'currently' reflects the passenger count after the current stop.
     * Invariant: 'total' always holds the highest passenger count encountered so far.
     */
    if (currently > total)
      total = currently;
  }
  // Output the maximum capacity required for the tram.
  cout << total << "
";
}
