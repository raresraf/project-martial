/**
 * @file 116A.CPP
 * @brief This program calculates the maximum capacity needed for a tram.
 * It simulates the entry and exit of passengers at various stops.
 * The current number of passengers in the tram is tracked ('total'),
 * and the maximum value achieved during the journey ('mx') is the required capacity.
 *
 * Algorithm: Iterative simulation of tram stops to find peak occupancy.
 * Time Complexity: O(N) where N is the number of tram stops.
 * Space Complexity: O(1)
 */

#include <bits/stdc++.h> // Includes iostream, etc.
using namespace std;

int main() {
  // Functional Utility: 'n' stores the total number of tram stops.
  int n;
  // Functional Utility: 'enter' stores the number of passengers entering at a stop.
  int enter;
  // Functional Utility: 'exit' stores the number of passengers exiting at a stop.
  int exit;
  // Functional Utility: 'total' tracks the current number of passengers in the tram.
  int total = 0;
  // Functional Utility: 'mx' (maximum) stores the maximum number of passengers observed in the tram.
  //                     Initialized to a very small number to ensure any positive passenger count will be larger.
  int mx = -999999;

  // Block Logic: Reads the total number of tram stops 'n' from standard input.
  cin >> n;

  /**
   * Block Logic: Iterates through each tram stop to simulate passenger changes and find peak occupancy.
   * Invariant: 'i' represents the current stop number, from 0 to n-1.
   */
  for (int i = 0; i < n; i++) {
    // Block Logic: Reads the number of passengers exiting ('exit') and entering ('enter') at the current stop.
    cin >> exit >> enter;
    // Functional Utility: Updates the current number of passengers in the tram by adding entries.
    total = total + enter;
    // Functional Utility: Updates the current number of passengers in the tram by subtracting exits.
    total = total - exit;
    /**
     * Block Logic: Checks if the current number of passengers ('total') exceeds the
     * previously recorded maximum ('mx').
     * Pre-condition: 'total' reflects the passenger count after the current stop.
     * Invariant: 'mx' always holds the highest passenger count encountered so far.
     */
    if (total > mx) {
      mx = total;
    }
  }
  // Output the maximum capacity required for the tram.
  cout << mx;
}
