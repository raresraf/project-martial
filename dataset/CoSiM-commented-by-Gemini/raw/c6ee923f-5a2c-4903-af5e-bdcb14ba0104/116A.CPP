/**
 * @file raw/c6ee923f-5a2c-4903-af5e-bdcb14ba0104/116A.CPP
 * @brief Calculates the minimum required capacity of a tram.
 *
 * This program simulates the passenger flow in a tram over a series of stops.
 * At each stop, some passengers exit and some enter. The goal is to determine
 * the maximum number of passengers on the tram at any one time, which corresponds
 * to the minimum capacity the tram must have.
 *
 * @algorithm
 * The algorithm iterates through each stop, maintaining a running count of the
 * current number of passengers.
 * 1. Initialize `current_passengers = 0` and `min_capacity = 0`.
 * 2. Loop `n` times for `n` stops.
 * 3. In each iteration, read the number of exiting passengers (`a`) and entering
 *    passengers (`b`).
 * 4. Update the passenger count: `current_passengers = current_passengers - a + b`.
 * 5. After each update, compare the `current_passengers` with `min_capacity` and
 *    update `min_capacity` if the current count is greater.
 * 6. After the loop, `min_capacity` holds the maximum passenger count observed.
 *
 * @time_complexity O(N), where N is the number of tram stops.
 * @space_complexity O(1).
 */
#include <iostream>
using namespace std;
int main() {
  int n, a, b;
  int count = 0, max; // `count` tracks current passengers, `max` tracks peak passengers.
  cin >> n;

  /**
   * @block
   * Main loop to process `n` tram stops.
   * Invariant: `count` holds the number of passengers on the tram after the
   * (i-1)-th stop, and `max` holds the peak number of passengers seen so far.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b; // `a` passengers exit, `b` passengers enter.
    
    // Update the current number of passengers on the tram.
    count -= a;
    count += b;
    
    // On the first stop, initialize `max` with the current passenger count.
    if (i == 0) {
      max = count;
    }

    // If the current passenger count exceeds the max seen so far, update max.
    if (max < count)
      max = count;
  }
  
  // Output the minimum required capacity.
  cout << max;
  return 0;
}
