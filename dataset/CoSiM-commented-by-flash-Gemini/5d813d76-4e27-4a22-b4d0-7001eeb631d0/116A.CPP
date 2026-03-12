/**
 * @5d813d76-4e27-4a22-b4d0-7001eeb631d0/116A.CPP
 * @brief This program solves problem 116A, commonly known as "Tram".
 * It simulates a tram's journey through 'n' stops. For each stop, it tracks
 * the number of passengers exiting and entering, calculates the current number
 * of passengers, and determines the maximum number of passengers observed
 * in the tram at any single moment.
 *
 * Algorithm: Linear simulation with passenger count tracking and maximum value update.
 * Time Complexity: O(N) where N is the number of tram stops.
 * Space Complexity: O(1) for storing variables.
 */

#include <iostream> // Includes the standard input/output stream library.
using namespace std; // Uses the standard namespace.

int main() { // Corrected main function signature to return int.
  int n; // Declares 'n', representing the number of tram stops.
  cin >> n; // Reads 'n' from standard input.
  int max = 0, no = 0; // Initializes 'max' (maximum passengers) and 'no' (current passengers) to 0.
  
  /**
   * Block Logic: Simulates the tram's journey through 'n' stops.
   * For each stop, it updates the current number of passengers and tracks the maximum observed.
   * Invariant: 'max' stores the highest passenger count encountered up to the current stop.
   */
  while (n > 0) { // Loops 'n' times, once for each stop.
    int a, b; // Declares 'a' for passengers exiting and 'b' for passengers entering.
    cin >> a >> b; // Reads passengers exiting and entering for the current stop.
    no = no + (b - a); // Inline: Updates the current number of passengers. `b - a` is the net change.
    // Inline: Updates 'max' if the current number of passengers 'no' exceeds the previously recorded maximum.
    if (max < no) 
      max = no;
    n -= 1; // Decrements the counter for remaining stops.
  }
  cout << max << endl; // Prints the maximum number of passengers observed, followed by a newline.
  return 0; // Indicates successful program termination.
}
