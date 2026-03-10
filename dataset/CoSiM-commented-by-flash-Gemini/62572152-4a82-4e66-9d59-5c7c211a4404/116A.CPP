/**
 * @file 116A.CPP
 * @brief This program solves the "Tram" problem (Codeforces 116A).
 *   It determines the minimum capacity a tram must have to accommodate all passengers
 *   at any point in time during its journey. Passengers enter and exit at `n` stops.
 *   Problem description: https://codeforces.com/problemset/problem/116/A
 * * Algorithm: Cumulative sum tracking.
 *   The program simulates the tram's journey, maintaining a running total of passengers
 *   currently inside the tram. At each stop, it updates this total and tracks the maximum
 *   number of passengers observed simultaneously, which represents the minimum required capacity.
 * Time Complexity: O(N) where N is the number of tram stops, as it involves a single pass through the stops.
 * Space Complexity: O(1) for a few integer variables.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.
#include <iostream>  // Required for std::cin, std::cout.
using namespace std;

/**
 * @brief Main function to calculate the minimum required tram capacity.
 * Functional Utility: Reads the number of stops and passenger changes at each stop,
 *                     simulates the tram's occupancy, and outputs the peak occupancy (minimum capacity).
 */
int main() {
  int n; // Stores the number of tram stops.
  // Block Logic: Reads the total number of stops.
  cin >> n;
  int a, b; // `a` for passengers exiting, `b` for passengers entering.
  int res = 0; // `res`: Tracks the current number of passengers inside the tram.
  int max_cap = -1; // `max_cap`: Stores the maximum number of passengers observed, initialized to -1.
  /**
   * Block Logic: Iterates through each of the `n` tram stops.
   * Pre-condition: `n` is a non-negative integer.
   * Invariant: In each iteration, `res` represents the current number of passengers,
   *            and `max_cap` stores the highest `res` encountered so far.
   * Inline: `n--` decrements the counter and evaluates to true as long as `n` is greater than 0.
   */
  while (n--) {
    // Block Logic: Reads the number of passengers exiting (`a`) and entering (`b`) at the current stop.
    cin >> a >> b;
    /**
     * Conditional Logic: Updates the current number of passengers (`res`).
     * `if (a == 0)` specifically handles the case where no passengers exit. This is functionally
     * equivalent to `res = res - a + b` if `a` is 0, but is explicitly separated here.
     * The `a` value is implicitly 0 at the first stop according to problem context, but other `a` can be non-zero.
     */
    if (a == 0)
      res = res + b;      // Passengers only enter.
    else
      res = res - a + b;  // Passengers exit then enter.
    // Conditional Logic: Updates `max_cap` to be the maximum of its current value and the current `res`.
    // This tracks the peak occupancy, which is the minimum required capacity.
    if (res > max_cap)
      max_cap = res;
  }
  // Block Logic: Prints the calculated minimum required capacity.
  cout << max_cap;
  return 0;
}
