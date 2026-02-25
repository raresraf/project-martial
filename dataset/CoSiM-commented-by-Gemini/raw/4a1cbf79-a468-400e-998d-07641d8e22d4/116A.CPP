/**
 * @file 116A.CPP
 * @brief Solution to Codeforces problem 116A (Tram).
 * @author Solved by: [User who submitted the code]
 *
 * @details
 * This program calculates the minimum required capacity of a tram based on the
 * number of passengers entering and exiting at each stop. The tram starts empty
 * and makes `n` stops.
 *
 * @section algorithm
 * The problem is solved by simulating the number of passengers on the tram at
 * each stop and finding the maximum value.
 *
 * The algorithm is as follows:
 * 1. Initialize `current_passengers` to 0 and `max_capacity` to 0.
 * 2. Loop through each of the `n` stops.
 * 3. For each stop, read the number of passengers exiting (`exit_count`) and
 *    entering (`enter_count`).
 * 4. Update the number of passengers on the tram:
 *    `current_passengers = current_passengers - exit_count + enter_count`.
 * 5. Update the required capacity:
 *    `max_capacity = max(max_capacity, current_passengers)`.
 * 6. After iterating through all stops, `max_capacity` will hold the minimum
 *    capacity the tram must have.
 *
 * The code implements this logic correctly.
 *
 * @section complexity
 * - Time Complexity: O(N), where N is the number of stops, because the code
 *   iterates through the stops once.
 * - Space Complexity: O(N) to store the passenger data for all stops in a 2D
 *   array. This could be optimized to O(1) by processing the data on the fly
 *   without storing the entire history, as the current state only depends on
 *   the previous state.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  // n: the number of tram stops.
  int n;
  cin >> n;

  // arr[n][2]: A 2D array to store passenger data for each stop.
  // arr[i][0] is the number of passengers exiting at stop i.
  // arr[i][1] is the number of passengers entering at stop i.
  int arr[n][2];
  for (int i = 0; i < n; i++)
    for (int j = 0; j < 2; j++)
      cin >> arr[i][j];

  // mx: Tracks the maximum number of passengers on the tram at any point.
  // sum: Tracks the current number of passengers on the tram.
  int mx = 0, sum = 0; // Initializing mx to 0 is safe as passenger count can't be negative.

  /**
   * @brief Loop to simulate the passenger flow at each stop.
   *
   * Invariant: `sum` holds the number of passengers on the tram as it leaves
   * the previous stop. `mx` holds the maximum value of `sum` encountered so far.
   */
  for (int i = 0; i < n; i++) {
    // Update the current number of passengers.
    // Passengers exit first, then new passengers enter.
    sum = sum - arr[i][0] + arr[i][1];

    // Update the maximum capacity required so far.
    mx = max(mx, sum);
  }

  // Output the minimum required capacity.
  cout << mx;

  return 0;
}
