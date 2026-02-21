/**
 * @file 116A.CPP
 * @brief Calculates the minimum capacity of a tram based on passenger flow at stops.
 * @author Andr s Sierra
 *
 * This program simulates the number of passengers on a tram over a series of stops
 * to determine the peak passenger count. This peak count represents the minimum
 * capacity the tram must have. The implementation handles the first stop as a
 * special case.
 *
 * Algorithm:
 * 1. Read the total number of stops, `n`.
 * 2. Initialize `capacity` (max passengers) and `temp` (current passengers).
 * 3. A `flag` variable stores the initial number of stops to identify the first
 *    iteration of the loop.
 * 4. Loop `n` times, once for each stop.
 *    a. Read the number of passengers exiting (`a`) and entering (`b`).
 *    b. For the first stop (`n == flag`):
 *       - The initial number of passengers is `b` (as `a` is 0).
 *       - Both `capacity` and `temp` are set to `b`.
 *    c. For all subsequent stops:
 *       - Update the current passenger count: `temp = (temp - a) + b`.
 *       - If the new count `temp` is greater than the max `capacity` recorded so far,
 *         update `capacity` to `temp`.
 * 5. After the loop, print the final `capacity`.
 *
 * Time Complexity: O(N), where N is the number of stops, due to the single loop.
 * Space Complexity: O(1), as it uses a fixed number of variables.
 */
#include <iostream>

using namespace std;

int main() {
  int n, capacity = 0;
  int a, b, temp;

  cin >> n;

  // `flag` is used to identify the first stop in the loop.
  int flag = n;

  /**
   * @brief Main loop to simulate passenger flow at each stop.
   * Pre-condition: `n` holds the number of stops.
   */
  while (n > 0) {
    cin >> a >> b;

    // Special case for the first stop.
    if (n == flag) {
      capacity = b;
      temp = b;
    }
    // Logic for all subsequent stops.
    else {
      // Update the current number of passengers.
      temp = (temp - a) + b;

      // If the current passenger count is the highest so far, update capacity.
      if (temp >= capacity)
        capacity = temp;
    }

    n--;
  }

  // Print the maximum capacity required.
  cout << capacity << endl;
}
