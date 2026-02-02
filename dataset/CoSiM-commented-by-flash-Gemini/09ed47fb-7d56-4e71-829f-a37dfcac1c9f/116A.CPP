/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A - Tram.
 *
 * This problem involves calculating the maximum capacity needed for a tram that traverses a route.
 * For each station, the number of passengers exiting and entering is given. The task is to find
 * the maximum number of passengers that are on the tram at any single moment.
 *
 * Algorithm: Iterative simulation.
 * The code simulates the tram journey by maintaining a `current_passengers` count.
 * For each station, it subtracts the passengers exiting and adds the passengers entering.
 * It also keeps track of the `maximum_passengers` encountered so far.
 *
 * Time Complexity: O(N) where N is the number of tram stops.
 * Space Complexity: O(1) as only a few variables are used to store state.
 */

#include <bits/stdc++.h> // Includes most standard C++ libraries, common in competitive programming.
using namespace std; // Functional Utility: Brings all identifiers from the std namespace into the current scope, simplifying code by avoiding explicit std:: prefixes.

/**
 * @brief Main function: Solves the Tram problem.
 * Functional Utility: Reads the number of tram stops, then for each stop, reads
 * the number of passengers exiting and entering. It calculates and prints the
 * maximum number of passengers on the tram at any point.
 * Algorithm: Iterative simulation of tram occupancy.
 */
int main() {
  int n, pas = 0, maks = 0; // n: number of stops, pas: current passengers, maks: maximum passengers.
  cin >> n; // Reads the number of tram stops.
  /**
   * Block Logic: Loop through each tram stop.
   * Precondition: `n` is a positive integer representing the number of stops.
   * Invariant: `pas` holds the current number of passengers, and `maks` holds the maximum encountered so far.
   *            The loop runs exactly `n` times, processing one stop per iteration.
   */
  while (n--) {
    int a, b; // a: passengers exiting, b: passengers entering.
    cin >> a >> b; // Reads the number of passengers exiting and entering for the current stop.
    pas -= a; // Decreases current passengers by those exiting.
    pas += b; // Increases current passengers by those entering.
    /**
     * Block Logic: Update the maximum passengers if the current passenger count exceeds it.
     * Invariant: `maks` always stores the highest passenger count recorded up to the current stop.
     */
    if (pas > maks)
      maks = pas; // Updates `maks` if `pas` is greater.
  }
  cout << maks << "\n"; // Prints the maximum number of passengers encountered.
  return 0; // Indicates successful program execution.
}