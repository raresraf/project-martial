/**
 * @file 116A.CPP
 * @brief This program simulates a tram journey to determine the maximum number of
 *        passengers present in the tram at any single moment.
 *
 * Algorithm:
 * 1. Read the total number of tram stops, 's'.
 * 2. Initialize 'max' to 0 (to track the peak passenger count) and 'current' to 0 (for the live passenger count).
 * 3. Loop 's' times, once for each tram stop:
 *    a. Read the number of passengers exiting ('l') and entering ('e') at the current stop.
 *    b. Update the 'current' passenger count: `current = current - l + e`.
 *    c. If 'current' is greater than or equal to 'max', update 'max' to 'current'.
 * 4. Print the final 'max' value, which represents the maximum capacity needed.
 *
 * Time Complexity: O(S) where S is the number of stops, as the program iterates through each stop once.
 * Space Complexity: O(1) as only a few integer variables are used.
 */

#include <iostream>
using namespace std;
/**
 * @brief Entry point of the program.
 * Functional Utility: Orchestrates the simulation by reading the number of stops,
 * processing passenger changes at each stop, and determining the maximum capacity reached.
 * @return 0 on successful execution.
 */
int main() {
  // Block Logic: Declares integer variables for the number of stops 's',
  //              passengers exiting 'e', passengers entering 'l',
  //              'max' for maximum capacity reached, and 'current' for current passengers.
  // Note: 'e' is used for exiting and 'l' for entering in this implementation.
  int s = 0, e = 0, l = 0, max = 0, current = 0;
  // Block Logic: Reads the total number of stops 's' from standard input.
  cin >> s;
  /**
   * Block Logic: Iterates 's' times, once for each tram stop, to update
   * the passenger count and track the maximum capacity.
   *
   * Invariant: At the start of each iteration, 's' represents the remaining stops,
   * 'current' holds the number of passengers currently in the tram, and 'max' stores
   * the highest passenger count encountered so far.
   */
  while (s--) {
    // Block Logic: Reads the number of passengers exiting ('l') and entering ('e') at the current stop.
    cin >> l >> e;
    // Block Logic: Updates the 'current' number of passengers by subtracting exiting and adding entering passengers.
    current = current + (e - l);
    // Block Logic: Updates 'max' if the 'current' passenger count exceeds the previously recorded maximum.
    if (current >= max)
      max = current;
  }
  // Block Logic: Prints the final calculated maximum number of passengers that were in the tram.
  cout << max;
}
