/**
 * @file 116A.CPP
 * @brief Solution for a simulation problem, likely from a programming contest.
 * @details The problem is to calculate the minimum required capacity of a tram based on
 * the number of passengers entering and exiting at each stop.
 *
 * Algorithm:
 * 1. Initialize `tram_passengers` to 0, representing the number of people currently on the tram.
 * 2. Initialize `max_capacity` to 0, which will store the peak number of passengers observed.
 * 3. Loop for 'n' stops:
 *    a. Read the number of passengers exiting (`a`) and entering (`b`).
 *    b. Update the current number of passengers: `tram_passengers = tram_passengers - a + b`.
 *    c. After the update, if `tram_passengers` is greater than `max_capacity`, update
 *       `max_capacity` to this new peak value.
 * 4. After simulating all stops, `max_capacity` will hold the minimum capacity the tram
 *    must have.
 *
 * Time Complexity: O(n), where 'n' is the number of stops.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, tram = 0, a, b, max = 0; // 'max' shadows std::max, but is used for capacity
  cin >> n;
  
  /**
   * @block
   * @description This loop simulates the passenger flow at each of the 'n' tram stops.
   * Invariant: At the end of each iteration, `tram` holds the current number of passengers,
   * and `max` holds the maximum number of passengers seen so far.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    // Update the number of passengers on the tram.
    tram = tram - a + b;
    
    // Check if the current passenger count is the new maximum.
    if (tram > max) {
      max = tram;
    }
  }
  
  // Output the minimum required capacity.
  cout << max;

  return 0;
}