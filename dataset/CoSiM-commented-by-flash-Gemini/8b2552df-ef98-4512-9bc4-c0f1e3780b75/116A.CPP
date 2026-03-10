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

// Functional Utility: Macro for a `for` loop with explicit start and end.
#define FOR(i, a, n) for (int i = (int)(a); i < (int)(n); ++i)
// Functional Utility: Macro for a `for` loop from 0 to `n-1`.
#define REP(i, n) FOR(i, 0, n)
// Functional Utility: Macro for getting iterators to the beginning and end of a container.
#define all(a) a.begin(), a.end()
// Functional Utility: Alias for `push_back` method of vectors.
#define pb push_back

// Functional Utility: Alias for `unsigned long long`.
typedef unsigned long long llu;
// Functional Utility: Alias for `long long`.
typedef long long ll;
// Functional Utility: Alias for `long double`.
typedef long double ld;

using namespace std;

/**
 * @brief Main function to calculate the minimum required tram capacity.
 * Functional Utility: Reads the number of stops and passenger changes at each stop,
 *                     simulates the tram's occupancy, and outputs the peak occupancy (minimum capacity).
 */
int main() {
  int curr = 0; // `curr`: Tracks the current number of passengers inside the tram.
  int n;        // `n`: Stores the number of tram stops.
  int m = 0;    // `m`: Stores the maximum number of passengers observed (peak occupancy).

  // Block Logic: Reads the total number of stops `n` using `scanf`.
  scanf("%d", &n);
  /**
   * Block Logic: Iterates through each of the `n` tram stops.
   * Pre-condition: `n` is a non-negative integer.
   * Invariant: In each iteration, `curr` represents the current number of passengers,
   *            and `m` stores the highest `curr` encountered so far.
   * Inline: `n--` decrements the counter and evaluates to true as long as `n` is greater than 0.
   */
  while (n--) {
    int a, b; // `a` for passengers exiting, `b` for passengers entering.
    // Block Logic: Reads the number of passengers exiting (`a`) and entering (`b`) at the current stop.
    scanf("%d %d", &a, &b);
    // Functional Utility: Updates the current number of passengers (`curr`).
    // Passengers exiting are subtracted, entering are added.
    curr -= a;
    curr += b;
    // Functional Utility: Updates `m` to be the maximum of its current value and the current `curr`.
    // This tracks the peak occupancy, which is the minimum required capacity.
    m = max(curr, m);
  }
  // Block Logic: Prints the calculated minimum required capacity.
  cout << m << endl;
  return 0;
}
