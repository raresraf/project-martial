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
#include "bits/stdc++.h" // Includes most standard C++ libraries.

using namespace std;

/**
 * @brief Main function to calculate the minimum required tram capacity.
 * Functional Utility: Reads the number of stops and passenger changes at each stop,
 *                     simulates the tram's occupancy, and outputs the peak occupancy (minimum capacity).
 */
int main(int argc, char const *argv[]) {
  int n; // Stores the number of tram stops.
  // Block Logic: Reads the total number of stops.
  cin >> n;

  int a, b; // `a` for passengers exiting, `b` for passengers entering.

  int max_cap = 0; // `max_cap`: Stores the maximum number of passengers observed, initialized to 0.
  int sum = 0;     // `sum`: Tracks the current number of passengers inside the tram, initialized to 0.

  /**
   * Block Logic: Iterates through each of the `n` tram stops.
   * Pre-condition: `n` is a non-negative integer.
   * Invariant: In each iteration, `sum` represents the current number of passengers,
   *            and `max_cap` stores the highest `sum` encountered so far.
   */
  for (int i = 0; i < n; i++) {
    // Block Logic: Reads the number of passengers exiting (`a`) and entering (`b`) at the current stop.
    cin >> a >> b;
    // Functional Utility: Updates the current number of passengers (`sum`). Passengers exiting are subtracted, entering are added.
    sum = sum - a + b;
    // Conditional Logic: Updates `max_cap` to be the maximum of its current value and the current `sum`.
    // This tracks the peak occupancy, which is the minimum required capacity.
    if (sum > max_cap)
      max_cap = sum;
  }

  // Block Logic: Prints the calculated minimum required capacity.
  cout << max_cap;
  return 0;
}
