/**
 * @file 116A.cpp
 * @brief Solution to a competitive programming problem: Tram.
 * This program calculates the minimum capacity of a tram required to accommodate
 * passengers over a series of stops. It simulates the passenger flow at each stop
 * and finds the maximum number of passengers inside the tram at any given time.
 *
 * Algorithm:
 * 1. Read the number of tram stops `n`.
 * 2. Initialize a variable `current_passengers` to 0 and `max_passengers` to 0.
 * 3. Loop `n` times, for each stop:
 *    a. Read the number of passengers getting off (`x`) and getting on (`y`).
 *    b. Update the number of passengers: `current_passengers = current_passengers - x + y`.
 *    c. Update `max_passengers` if `current_passengers` is greater than the current `max_passengers`.
 * 4. Print `max_passengers`.
 *
 * Time Complexity: O(N), where N is the number of stops.
 * Space Complexity: O(1) (the provided solution uses an array which is not necessary, an optimized solution would have O(1) space).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // n: number of stops, i: loop counter, x: passengers leaving, y: passengers entering.
  // entery: current passengers on the tram, leaving: passengers on tram after some leave.
  // m: maximum passengers at any time, initialized to a very small number.
  int n, i, x, y, a[1000], entery = 0, leaving = 0;
  int m = -32760;
  // Read the number of stops.
  cin >> n;
  
  /**
   * @brief Loop through each stop to calculate passenger count.
   * `x` passengers get off, `y` passengers get on.
   * `entery` keeps track of the number of people inside the tram.
   */
  for (i = 0; i < n; i++) {
    cin >> x >> y;
    // Special handling for the first stop.
    if (i == 0) {
      entery = y - x;
    } else {
      // For subsequent stops, calculate passengers after some leave, then add new ones.
      leaving = entery - x;
      entery = leaving + y;
    }
    // Store the passenger count at each stop in an array.
    a[i] = entery;
  }

  /**
   * @brief Find the maximum value in the array of passenger counts.
   * This represents the minimum capacity required for the tram.
   */
  for (i = 0; i < n; i++) {
    if (m < a[i]) {
      m = a[i];
    }
  }
  // Print the maximum capacity.
  cout << m << endl;

  return 0;
}
