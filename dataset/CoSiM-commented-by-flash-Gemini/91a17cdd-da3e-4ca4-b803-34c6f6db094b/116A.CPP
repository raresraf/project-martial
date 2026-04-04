/**
 * @file 116A.CPP
 * @brief Calculates the maximum capacity required for a tram using a single running counter.
 *
 * This program simulates a tram journey with 'n' stops. At each stop, it records
 * the number of passengers who exit and the number who enter. The objective is
 * to determine the maximum number of passengers present on the tram at any point
 * throughout its journey. This version uses a single counter 'c' for current passengers
 * and 'm' for maximum passengers.
 *
 * Algorithm: Iterative simulation with optimized state tracking. Maintain a running
 *            total of passengers on the tram and continuously update the maximum
 *            number observed.
 * Time Complexity: O(N), where N is the number of stops, as each stop is processed once.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h> // Includes most standard library headers.

using namespace std;

#define ll long long // Functional Utility: Defines 'll' as a shorthand for 'long long'.

int main() {
  /**
   * @brief Main function: Simulates tram journey to find maximum passenger count.
   * @return 0 if the program executes successfully.
   */
  int n; // Number of stops.
  cin >> n; // Read the number of stops.
  int c = 0; // 'c': Current number of people on the tram.
  int m = 0; // 'm': Maximum number of people observed on the tram.
  int a, b; // 'a': Passengers leaving, 'b': Passengers entering.
  /**
   * @brief Loop through each stop of the tram journey.
   * @pre 'n' is a positive integer. 'c' and 'm' are initialized to 0.
   * @post 'm' holds the maximum number of people on the tram throughout the journey.
   * @invariant 'c' accurately reflects the number of people on the tram after each stop.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b; // Read the number of passengers leaving ('a') and entering ('b').
    c -= a; // Passengers leave the tram.
    c += b; // Passengers enter the tram.
    /**
     * @brief Conditional block to update the maximum number of people observed on the tram.
     * @pre 'c' reflects the current number of passengers.
     */
    if (c > m)
      m = c; // Update 'm' if the current passenger count is a new maximum.
  }
  cout << m << "\n"; // Output the maximum number of people observed on the tram.
  return 0; // Return 0 for successful execution.
}
