/**
 * @file 116A.CPP
 * @brief This program calculates the maximum capacity needed for a tram.
 * It simulates the entry and exit of passengers at various stops.
 * The current number of passengers in the tram is tracked, and the maximum
 * value achieved during the journey is the required capacity.
 *
 * Algorithm: Iterative simulation of tram stops to find peak occupancy.
 * Time Complexity: O(L) where L is the number of tram stops.
 * Space Complexity: O(L) to store the passenger counts for each stop.
 */

#include <cmath>   // Required for std::max (though <algorithm> is usually preferred for it)
#include <iostream>
using namespace std;

int main() {
  // Functional Utility: 'l' stores the total number of tram stops.
  int l;
  // Block Logic: Reads the total number of tram stops 'l' from standard input.
  cin >> l;

  // Functional Utility: 'A' is an array to store the number of passengers exiting at each stop.
  int A[1000];
  // Functional Utility: 'B' is an array to store the number of passengers entering at each stop.
  int B[1000];
  /**
   * Block Logic: Reads the passenger changes (exiting and entering) for each of the 'l' stops.
   * Invariant: After each iteration, `A[i]` and `B[i]` hold the exit and entry counts for stop 'i'.
   */
  for (int i = 0; i < l; i++) {
    cin >> A[i] >> B[i];
  }

  // Functional Utility: 'a' tracks the current number of passengers in the tram.
  int a = 0;
  // Functional Utility: 'b' stores the maximum number of passengers observed in the tram (peak capacity).
  int b = 0;

  /**
   * Block Logic: Iterates through each tram stop to simulate passenger changes and find peak occupancy.
   * Invariant: After each iteration, 'a' reflects the current passenger count, and 'b' holds
   *            the maximum passenger count encountered up to that stop.
   */
  for (int i = 0; i < l; i++) {
    // Functional Utility: Updates the current number of passengers in the tram.
    // Inline: Passengers exiting are subtracted, and passengers entering are added.
    a = (a - A[i]) + B[i];
    // Functional Utility: Updates the maximum observed passenger count if the current count is higher.
    b = max(a, b);
  }

  // Output the maximum capacity required for the tram.
  cout << b << endl;
}
