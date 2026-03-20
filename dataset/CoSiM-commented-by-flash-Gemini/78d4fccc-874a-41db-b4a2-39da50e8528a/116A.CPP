/**
 * @file 116A.CPP
 * @brief Solution to Codeforces problem 116A - Tram.
 *        This program simulates the passenger count within a tram as it moves between stops.
 *        For each stop, it records the number of passengers exiting and entering, and
 *        determines the maximum number of passengers simultaneously present in the tram.
 * Algorithm:
 *   1. Initialize `sum` (current passengers) and `mx` (maximum passengers) to 0.
 *   2. Read the total number of stops, `n`.
 *   3. For each stop (from 0 to n-1):
 *      a. Read the number of passengers exiting (`a`) and entering (`b`).
 *      b. If it's the first stop, the initial passenger count (`sum`) is just `b`, and `mx` is also `b`.
 *         The loop then continues to the next iteration.
 *      c. For subsequent stops, subtract `a` from `sum` (passengers exit).
 *      d. Add `b` to `sum` (passengers enter).
 *      e. Update `mx` if `sum` is greater than the current `mx`.
 *   4. After all stops are processed, print the final `mx` value.
 * Time Complexity: O(n), where n is the number of stops, as each stop is processed once.
 * Space Complexity: O(1) as only a few variables are used to store counts and sums.
 */
#include <iostream>

using namespace std;

int main() {
  /**
   * @brief Main function implementing the solution for the "Tram" problem.
   *
   * Simulates the passenger count on a tram, updating the current number of passengers
   * at each stop and tracking the maximum number of passengers observed.
   * @return int Returns 0 upon successful execution.
   */
  int n, a, b, i, mx, sum;

  cin >> n; // Read the number of tram stops.

  // Block Logic: Iterates through each tram stop to simulate passenger movement.
  // Pre-condition: 'n' is the total number of stops.
  // Invariant: 'sum' tracks current passengers, 'mx' tracks max passengers observed.
  for (i = 0; i < n; i++) {
    cin >> a >> b; // Read passengers exiting ('a') and entering ('b') at the current stop.

    // Block Logic: Special handling for the first stop to initialize sum and maximum.
    // Pre-condition: This is the first iteration (i == 0).
    // Post-condition: 'sum' and 'mx' are initialized with the number of passengers entering the first stop.
    if (i == 0) {
      sum = b; // Initialize current passenger count with those entering the first stop.
      mx = b;  // The maximum passengers so far is also this initial count.
      continue; // Skip the rest of the loop for the first iteration as sum is already initialized.
    }

    sum -= a; // Passengers 'a' exit the tram.
    sum += b; // Passengers 'b' enter the tram.
    mx = max(mx, sum); // Update maximum passengers if current sum is higher.
  }

  cout << mx; // Output the maximum number of passengers observed at any point.

  return 0;
}
