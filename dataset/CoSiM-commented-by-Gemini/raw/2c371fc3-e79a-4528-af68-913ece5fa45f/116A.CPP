/**
 * @file 116A.CPP
 * @brief Solution to a problem calculating the minimum capacity of a tram.
 * @details This program determines the minimum possible capacity a tram could have to accommodate
 * passengers over a series of stops. It tracks the number of passengers inside the tram at each point
 * and finds the maximum number of passengers present at any one time.
 *
 * Problem description:
 * https://codeforces.com/problemset/problem/116/A
 *
 * Algorithm:
 * 1. Read the total number of stops, `n`.
 * 2. Initialize `p` (current passengers) and `mp` (maximum passengers) to 0.
 * 3. Loop `n` times, once for each stop.
 * 4. At each stop, read the number of exiting passengers `a` and entering passengers `b`.
 * 5. Update the current number of passengers: `p = p - a + b`.
 * 6. After updating, if the current number of passengers `p` is greater than the maximum `mp` recorded so far, update `mp` to `p`.
 * 7. After the loop finishes, `mp` will hold the peak passenger count, which is the minimum required capacity of the tram.
 * 8. Print the value of `mp`.
 *
 * Time Complexity: O(n), where 'n' is the number of stops, because the program iterates through the stops once.
 * Space Complexity: O(1), as it uses a fixed number of variables.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  short n, a, b, p(0), mp(0);
  // n: number of tram stops
  // a: passengers exiting
  // b: passengers entering
  // p: current passengers on the tram, initialized to 0
  // mp: maximum passengers seen so far (minimum capacity), initialized to 0
  cin >> n;

  /**
   * Block Logic: Simulate the passenger flow at each stop to find the peak occupancy.
   * Pre-condition: 'n' is the number of stops.
   * Invariant: After each iteration, 'p' holds the passenger count after that stop, and 'mp' holds the maximum value of 'p' seen up to that point.
   */
  for (short i = 0; i < n; i++) {
    cin >> a >> b;
    p -= a; // Passengers exit.
    p += b; // Passengers enter.
    
    // If the current number of passengers is the highest yet, update the max capacity.
    if (p > mp)
      mp = p;
  }
  // Output the minimum required capacity.
  cout << mp << '\n';
}
