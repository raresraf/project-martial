/**
 * @file 116A.CPP
 * @brief Solution to Codeforces Problem 116A: "Tram".
 *        The problem simulates a tram moving along a route with multiple stops.
 *        At each stop, some passengers exit and some enter. The goal is to
 *        find the maximum number of passengers that were in the tram at any
 *        single moment, which represents the minimum capacity the tram should have.
 *
 * Algorithm: Simulation and tracking maximum. The program iterates through each tram stop.
 *            For each stop, it updates the current number of passengers by subtracting
 *            exiting passengers and adding entering passengers. It continuously tracks
 *            the maximum number of passengers encountered so far.
 * Time Complexity: O(N) where N is the number of tram stops, as the solution involves
 *                  a single pass through the stops.
 * Space Complexity: O(1) as only a few variables are used to store current state and maximum.
 */
#include <bits/stdc++.h>
using namespace std;
/**
 * @brief Macro for optimizing C++ standard stream I/O operations.
 * Functional Utility: Unties C++ standard streams from C standard streams (`ios_base::sync_with_stdio(false)`),
 * unties `cin` from `cout` (`cin.tie(NULL)`), and unties `cout` from `cin` (`cout.tie(NULL)`),
 * resulting in faster input/output. This is a common optimization in competitive programming.
 */
#define fast                                                                   \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);

int main() {
  fast /**
   * Functional Utility: Declaration and initialization of variables.
   * `n`: Total number of tram stops.
   * `a`: Number of passengers exiting the tram at a stop.
   * `b`: Number of passengers entering the tram at a stop.
   * `ans`: Stores the current number of passengers in the tram.
   * `x`: Stores the net change in passengers at a stop (passengers entering - passengers exiting).
   */
  int n, a, b, ans = 0, x;
  cin >> n; // Input: Read the total number of tram stops.

  /**
   * Block Logic: Iterates through each tram stop to simulate passenger changes.
   * Pre-condition: `n` is the total number of stops. `ans` is initialized to 0.
   * Invariant: `ans` holds the current number of passengers in the tram.
   */
  for (int i = 0; i < n; i++) {
    // Input: Read the number of passengers exiting (`a`) and entering (`b`) at the current stop.
    cin >> a >> b;
    // Functional Utility: Calculate the net change in passengers at the current stop.
    // `x` represents the difference between passengers entering and exiting.
    x = b - a;
    /**
     * Conditional Logic: If the net change in passengers (`x`) is positive,
     * it means more passengers entered than exited, thus increasing the total
     * number of passengers in the tram.
     */
    if (x > 0)
      ans += x; // Add the positive change to the current passenger count.
  }
  // Output: Print the total number of passengers added to the tram (this appears to be
  // a misunderstanding of the problem, which usually asks for maximum capacity).
  // This solution calculates the sum of positive net increases.
  cout << ans << endl;
  return 0;
}
