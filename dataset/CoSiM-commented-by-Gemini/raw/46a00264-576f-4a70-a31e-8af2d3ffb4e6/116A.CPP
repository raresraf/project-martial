/**
 * @file raw/46a00264-576f-4a70-a31e-8af2d3ffb4e6/116A.CPP
 * @brief Solution to Codeforces problem 116A - Tram.
 *
 * @algorithm
 * The problem requires finding the minimum possible capacity of a tram, given the
 * number of passengers exiting and entering at each of `n` stops. The capacity
 * must be at least the maximum number of passengers on the tram at any given time.
 *
 * The solution is a direct simulation of the process:
 * 1. Initialize `current_passengers` to 0. This variable tracks the number of
 *    people inside the tram.
 * 2. Initialize `min_capacity` to 0. This variable will store the maximum value
 *    that `current_passengers` reaches during the simulation.
 * 3. Loop through each of the `n` stops.
 * 4. At each stop, read the number of passengers exiting (`exit_count`) and
 *    entering (`enter_count`).
 * 5. Update the passenger count:
 *    `current_passengers = current_passengers - exit_count + enter_count`.
 * 6. After updating, compare `current_passengers` with `min_capacity`. If the
 *    current count is higher, update `min_capacity` to this new maximum.
 * 7. After the loop finishes, `min_capacity` holds the minimum capacity required
 *    for the entire trip. This value is then printed.
 *
 * @complexity
 * - Time Complexity: O(n), where `n` is the number of stops, due to the single
 *   loop through all the stops.
 * - Space Complexity: O(1), as the simulation only requires a few variables to
 *   keep track of the state.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n; // Number of stops.
  int tram = 0; // Current number of passengers on the tram.
  int a, b; // a: passengers exiting, b: passengers entering.
  
  // `max` stores the maximum number of passengers seen so far.
  // It's not good practice to name a variable `max` as it can clash with `std::max`.
  int max = 0; 
  
  cin >> n;

  /**
   * @brief Simulate the passenger flow at each stop.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    // Update the number of passengers: subtract exits, add entrances.
    tram = tram - a + b;
    // If the current number of passengers is a new maximum, update the capacity tracker.
    if (tram > max) {
      max = tram;
    }
  }

  // Print the minimum required capacity.
  cout << max;

  return 0;
}
