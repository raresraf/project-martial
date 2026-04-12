/**
 * @file 116A.CPP
 * @brief Solution to the "Tram" problem.
 *
 * This program simulates the passenger flow in a tram to determine the maximum
 * number of passengers present in the tram at any point in time. At each stop,
 * passengers exit and then passengers enter. The tram's current occupancy is
 * tracked, and the maximum occupancy encountered is recorded.
 *
 * Algorithm: Simulation with Max Tracking.
 * Time Complexity: O(N) where N is the number of tram stops, as each stop's
 *                  data is processed once.
 * Space Complexity: O(1) as only a few variables are used to track current
 *                   occupancy and maximum occupancy.
 */
#include <bits/stdc++.h>
#define ll long long int // Type alias for long long integer.
#define loop(k) for (i = 0; i < k; ++i) // Macro for a common loop pattern (from 0 to k-1).
#define loop2(k, l) for (j = k; j < l; ++j) // Macro for another common loop pattern (from k to l-1).
#define mod 1000000007 // Define a modulo constant, common in competitive programming.
using namespace std;
/**
 * @brief Main function to simulate tram passenger flow and find maximum occupancy.
 *
 * Functional Utility: Configures fast I/O, reads the number of stops, then
 * iteratively updates current passenger count and tracks the maximum.
 */
int main() {
  // Functional Utility: Disables synchronization between C-style I/O and C++ streams,
  // and unties cin from cout, significantly speeding up input/output operations.
  std::ios_base::sync_with_stdio(false); // cin.tie(NULL); - commented out, as cin.tie(NULL) implies this.
  int n, i; // 'n' for number of stops, 'i' for loop counter.
  cin >> n; // Read the number of tram stops.
  int a, b, count = 0, d = 0; // 'a' for passengers exiting, 'b' for passengers entering,
                              // 'count' for current passengers, 'd' for maximum passengers.

  /**
   * Block Logic: Iterates through each tram stop, processing passenger exits and entries.
   * Updates the current number of passengers and tracks the maximum ever present.
   * Pre-condition: 'n' (number of stops) has been read.
   * Invariant: 'count' always holds the current number of passengers in the tram,
   *            and 'd' always holds the maximum 'count' observed so far.
   */
  loop(n) { // Loop 'n' times, once for each stop.
    cin >> a >> b; // Read passengers exiting (a) and entering (b).
    count += (b - a); // Update current passenger count: add entries, subtract exits.
    d = max(d, count); // Update maximum passenger count if current count is higher.
  }
  cout << d; // Output the maximum number of passengers observed.
  cout << "\n"; // Print a newline character.
  return 0;
}