/**
 * @file 116A.CPP
 * @brief This program solves the "Tram" problem (Codeforces 116A).
 *   It determines the minimum capacity a tram must have to accommodate all passengers
 *   at any point in time during its journey. Passengers enter and exit at `n` stops.
 *   Problem description: https://codeforces.com/problemset/problem/116/A
 * * Algorithm: Cumulative sum (prefix sum) tracking.
 *   The program simulates the tram's journey, maintaining a running total of passengers
 *   currently inside the tram. At each stop, it updates this total and tracks the maximum
 *   number of passengers observed simultaneously, which represents the minimum required capacity.
 * Time Complexity: O(N) where N is the number of tram stops, as it involves a single pass through the stops.
 * Space Complexity: O(1) for a few integer variables.
 */
#include <iostream>  // Required for standard input/output operations.
#include <algorithm> // Required for std::max.
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
  int minCap = 0; // Stores the minimum required capacity, effectively the maximum passenger count observed.
  int a, b;       // `a` for passengers exiting, `b` for passengers entering.
  int sum = 0;    // Tracks the current number of passengers inside the tram.
  /**
   * Block Logic: Iterates through each of the `n` tram stops.
   * Pre-condition: `n` is a non-negative integer.
   * Invariant: In each iteration, `sum` represents the current number of passengers,
   *            and `minCap` stores the highest `sum` encountered so far.
   */
  for (int i = 0; i < n; ++i) {
    // Block Logic: Reads the number of passengers exiting (`a`) and entering (`b`) at the current stop.
    cin >> a >> b;
    // Functional Utility: Updates the current number of passengers. Passengers exiting are subtracted, entering are added.
    sum += (b - a);
    // Functional Utility: Updates `minCap` to be the maximum of its current value and the current `sum`.
    // This finds the peak occupancy, which directly corresponds to the minimum required capacity.
    minCap = max(minCap, sum);
  }
  // Block Logic: Prints the calculated minimum required capacity.
  cout << minCap;
  return 0;
}
