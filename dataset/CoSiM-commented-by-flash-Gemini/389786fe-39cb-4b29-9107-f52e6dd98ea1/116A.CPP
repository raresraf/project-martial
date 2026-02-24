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

#include <cstdio> // For standard input/output functions like scanf and printf.
using namespace std; // Uses the standard namespace.

int main() {
  /**
   * @brief Main function to execute the Tram problem logic.
   *
   * Reads the number of tram stops 'n'. Then, for each stop, it reads the
   * number of passengers who exit ('a') and enter ('b'). It continuously
   * updates the current number of passengers in the tram ('x') and keeps
   * track of the maximum number of passengers encountered at any point
   * ('capacity'). Finally, it prints this maximum passenger count.
   */
  // Declare variables:
  // 'n' for the number of tram stops.
  // 'a' for passengers exiting at a stop.
  // 'b' for passengers entering at a stop.
  // 'x' to store the current number of passengers in the tram, initialized to 0.
  // 'capacity' to store the maximum number of passengers observed, initialized to 0.
  int n, a, b, x = 0, capacity = 0;
  // Read the number of tram stops.
  scanf("%d", &n);
  // Block Logic: Process each tram stop.
  // Invariant: 'n' decreases with each iteration, processing one stop's data per loop.
  while (n--) {
    // Read the number of passengers exiting and entering at the current stop.
    scanf("%d%d", &a, &b);
    // Update the current number of passengers in the tram: subtract exits, add entries.
    x = x - a + b;
    // If the current passenger count 'x' exceeds the previously recorded maximum 'capacity',
    // update 'capacity' with the new maximum.
    if (x > capacity)
      capacity = x;
  }
  // Print the maximum passenger count observed, which represents the minimum required capacity.
  printf("%d", capacity);
}
