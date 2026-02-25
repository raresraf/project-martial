/**
 * @file 116A.CPP
 * @brief Solution for the "Tram" capacity problem from a programming contest.
 * @details This program calculates the minimum required capacity of a tram by simulating
 * the change in passenger numbers over a series of stops.
 *
 * Algorithm:
 * The solution uses a single-pass approach to track the number of passengers.
 * 1. Initialize `current_passengers` and `max_capacity` to 0.
 * 2. Loop 'n' times for each stop.
 * 3. In each iteration, read the number of passengers exiting and entering.
 * 4. Update the `current_passengers` count by adding the entering passengers
 *    and subtracting the exiting ones.
 * 5. Compare the new `current_passengers` count with `max_capacity`. If it's
 *    larger, update `max_capacity`.
 * 6. After the loop completes, `max_capacity` will hold the peak passenger
 *    count observed, which is the minimum capacity required for the tram.
 *
 * Time Complexity: O(n), where 'n' is the number of stops.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  int n, enter, exit, result = 0, capasity = 0;

  cin >> n;

  /**
   * @block
   * @description This loop simulates the passenger flow over 'n' stops.
   * 'result' tracks the current number of people on the tram, and 'capasity'
   * tracks the maximum value 'result' has reached.
   */
  while (n--) {
    cin >> exit >> enter;

    // Update the current passenger count.
    result = result + (enter - exit);

    // If the current count is a new peak, update the required capacity.
    if (result > capasity) {
      capasity = result;
    }
  }

  // Output the minimum capacity required.
  cout << capasity;

  return 0;
}