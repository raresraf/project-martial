/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A - Tram.
 *
 * This program calculates the minimum capacity a tram needs to have to avoid
 * overcrowding, given a sequence of passenger exits and entries at each stop.
 *
 * Algorithm: Simulation and tracking maximum. The program simulates the passenger
 * flow through the tram. It keeps track of the current number of passengers
 * and updates a maximum value whenever the current passenger count exceeds it.
 * Time Complexity: O(N), where N is the number of tram stops.
 * Space Complexity: O(1), as only a few variables are stored.
 */

#include <stdio.h>  // For standard input/output functions like scanf and printf.
#include <string.h> // Included but not explicitly used in this solution.

int main() {
  /**
   * @brief Main function to execute the Tram problem logic.
   *
   * Reads the number of tram stops 'n'. Then, for each stop, it reads the
   * number of passengers who exit ('a') and enter ('b'). It continuously
   * updates the current number of passengers in the tram and keeps track of
   * the maximum number of passengers encountered at any point.
   * Finally, it prints this maximum passenger count.
   */
  // Declare variables:
  // 'i' for loop counter.
  // 'n' for the number of tram stops.
  // 'a' for passengers exiting at a stop.
  // 'b' for passengers entering at a stop.
  // 't1' (unused in this solution, likely leftover from template/draft).
  // 'temp' to store the maximum number of passengers observed so far.
  // 'in' to store the current number of passengers in the tram.
  int i, n, a, b, t1 = 0, temp = 0, in = 0;
  // Read the number of tram stops.
  scanf("%d", &n);
  // Block Logic: Process each tram stop.
  // Invariant: 'i' iterates 'n' times, updating passenger counts at each stop.
  for (i = 0; i < n; i++) {
    // Read the number of passengers exiting and entering at the current stop.
    scanf("%d %d", &a, &b);
    // Update the current number of passengers in the tram: subtract exits, add entries.
    in = b - a + in;
    // Check if the current passenger count ('in') is greater than the maximum observed so far ('temp').
    // The condition 'i == 0' ensures that 'temp' is initialized with the passenger count after the first stop.
    if (i == 0 || in > temp) {
      temp = in; // Update 'temp' if a new maximum is found.
    }
  }
  // Print the maximum passenger count observed, which represents the minimum required capacity.
  printf("%d", temp);

  return 0; // Indicate successful program execution.
}