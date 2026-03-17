/**
 * @file 116A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 116A "Tram".
 *
 * @details
 * This program calculates the minimum required capacity of a tram by finding the
 * maximum number of passengers on board at any point. It's given the number of
 * stops, and for each stop, the number of passengers exiting and entering.
 *
 * @algorithm
 * This is a space-efficient O(1) implementation. It simulates the process
 * stop-by-stop without storing the entire history of entries and exits.
 * 1. Initialize `current_passengers` and `max_capacity` to 0.
 * 2. Loop `n` times, once for each stop.
 * 3. In each loop, read the exiting and entering passenger counts for that stop.
 * 4. Update the `current_passengers` count.
 * 5. Update `max_capacity` if `current_passengers` is greater than the current max.
 * 6. After the loop, `max_capacity` holds the answer.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the number of stops, due to the single loop.
 * - Space Complexity: O(1), as it uses a fixed number of variables.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function and entry point of the program.
 */
int main() {
  int i, n, a, b;
  int cnt = 0; // Represents the current number of passengers on the tram.
  int sum = 0; // Represents the maximum number of passengers seen so far.

  // Read the total number of stops.
  cin >> n;

  /**
   * @brief Loop through each stop to simulate the passenger flow.
   */
  for (i = 0; i < n; i++) {
    // a: passengers exiting, b: passengers entering.
    cin >> a >> b;

    // Update the current passenger count.
    cnt = (cnt - a) + b;

    // If the current count is a new maximum, update the required capacity.
    if (cnt >= sum) {
      sum = cnt;
    }
  }
  
  // The highest recorded passenger count is the minimum capacity needed.
  cout << sum;
  return 0;
}
