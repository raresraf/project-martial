/**
 * @file 116A.CPP
 * @brief Solution for a discrete capacity simulation problem, often called the "Tram" problem.
 * @details This program calculates the minimum tram capacity required by tracking the number
 * of passengers at each stop. It simulates the change in passenger count and records the
 * maximum number of people on the tram at any point.
 *
 * @algorithm Iterative Simulation
 * @complexity
 * - Time: O(n), where 'n' is the number of stops, due to a single loop over the stops.
 * - Space: O(1), as it uses a fixed number of variables for the calculation.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // n: the number of tram stops.
  // a: passengers exiting at a stop.
  // b: passengers entering at a stop.
  // max: the maximum number of passengers recorded so far (i.e., the required capacity).
  // tot: the current total number of passengers on the tram.
  int n, a, b, max = 0, tot = 0;
  cin >> n;

  /**
   * @block Main Simulation Loop
   * @logic Iterates `n` times, once for each stop, to update the passenger count.
   */
  while (n--) {
    cin >> a >> b;
    // Update the current total by subtracting exiting passengers and adding entering ones.
    tot += -a + b;
    // If the current total exceeds the max recorded, update the max.
    if (max < tot)
      max = tot;
  }
  
  // Output the final calculated minimum capacity.
  cout << max << endl;
  return 0;
}
