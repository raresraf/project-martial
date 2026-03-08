/**
 * @file 116A.CPP
 * @brief Solution for a discrete capacity simulation problem, often framed as a "Tram" problem.
 *
 * This program calculates the maximum number of passengers inside a tram at any given time
 * across a series of stops.
 *
 * @algorithm Iterative Simulation
 * @complexity
 * - Time: O(n), where 'n' is the number of tram stops, as we iterate through each stop once.
 * - Space: O(1), as a constant amount of memory is used regardless of the input size.
 */

#include "bits/stdc++.h"

using namespace std;

int main(int argc, char const *argv[]) {
  // Number of tram stops.
  int n;
  cin >> n;

  // 'a' is the number of passengers exiting, 'b' is the number of passengers entering.
  int a, b;

  // 'max' stores the maximum passenger capacity required.
  int max = 0;
  // 'sum' tracks the current number of passengers on the tram.
  int sum = 0;

  /**
   * @block Main Simulation Loop
   * @logic Iterates through each stop to simulate passenger flow.
   * @invariant `sum` holds the number of passengers on the tram before the current stop's exchange.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    // Update the current passenger count by subtracting those who exit and adding those who enter.
    sum = sum - a + b;
    /**
     * @block Capacity Check
     * @logic If the current passenger count exceeds the previously recorded maximum, update the maximum.
     * @invariant `max` holds the highest value `sum` has reached in all previous iterations.
     */
    if (sum > max)
      max = sum;
  }

  // Output the minimum capacity required to handle the passenger load.
  cout << max;
  return 0;
}
