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
#include <cstdio> // Required for `scanf`, `printf`.
using namespace std;

/**
 * @brief Main function to calculate the minimum required tram capacity.
 * Functional Utility: Reads the number of stops and passenger changes at each stop,
 *                     simulates the tram's occupancy, and outputs the peak occupancy (minimum capacity).
 */
int main() {
  int n; // Stores the number of tram stops.
  // `a`: passengers exiting, `b`: passengers entering.
  // `x`: Tracks the current number of passengers inside the tram.
  // `capacity`: Stores the maximum number of passengers observed (peak occupancy).
  int a, b, x = 0, capacity = 0;
  // Block Logic: Reads the total number of stops `n` using `scanf`.
  scanf("%d", &n);
  /**
   * Block Logic: Iterates through each of the `n` tram stops.
   * Pre-condition: `n` is a non-negative integer.
   * Invariant: In each iteration, `x` represents the current number of passengers,
   *            and `capacity` stores the highest `x` encountered so far.
   * Inline: `n--` decrements the counter and evaluates to true as long as `n` is greater than 0.
   */
  while (n--) {
    // Block Logic: Reads the number of passengers exiting (`a`) and entering (`b`) at the current stop.
    scanf("%d%d", &a, &b);
    // Functional Utility: Updates the current number of passengers (`x`).
    // Passengers exiting are subtracted, and passengers entering are added.
    x = x - a + b;
    // Conditional Logic: Updates `capacity` to be the maximum of its current value and the current `x`.
    // This tracks the peak occupancy, which is the minimum required capacity.
    if (x > capacity)
      capacity = x;
  }
  // Block Logic: Prints the calculated minimum required capacity using `printf`.
  printf("%d", capacity);
}
