/**
 * @file 116A.CPP
 * @brief An incorrect attempt to solve the "Tram" problem from Codeforces.
 * @details The "Tram" problem (116A) requires finding the maximum number of
 * passengers on a tram at any single time to determine its minimum required
 * capacity. This code, however, does not correctly calculate this value.
 *
 * @algorithm (as implemented)
 * The logic implemented here is flawed. Instead of tracking the current number
 * of passengers and finding the maximum, it calculates the net change at each
 * stop and only accumulates the positive net changes. This does not yield the
 * maximum capacity required.
 * 1. Read the number of stops, n.
 * 2. Initialize a result variable `ans` to 0.
 * 3. Loop n times for each stop:
 *    a. Read the number of exiting (`a`) and entering (`b`) passengers.
 *    b. Calculate the net change `x = b - a`.
 *    c. If the net change `x` is positive, add it to `ans`.
 * 4. Print the final sum `ans`.
 *
 * @correct_algorithm
 * A correct solution would maintain a 'current_passengers' variable, updating
 * it with the net change at each stop (`current_passengers += b - a`), and
 * simultaneously track the maximum value that 'current_passengers' reaches.
 *
 * @complexity
 * Time Complexity: O(n), for the single loop over n stops.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

// Macro for fast I/O in competitive programming.
#define fast                                                                   \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);

int main() {
  fast 
  int n; // Number of stops.
  int a, b; // a: passengers exiting, b: passengers entering.
  int ans = 0; // The result variable.
  int x; // The net change in passengers at a stop.
  
  cin >> n;

  /**
   * @block
   * @brief Loop through each stop and incorrectly accumulate passenger changes.
   * @warning This logic is flawed for the "Tram" problem. It only sums the
   * net *increases* in passengers, rather than tracking the running total
   * to find the peak occupancy.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    x = b - a;
    // This only adds to the total when more people enter than exit. It does
    // not correctly model the tram's current capacity.
    if (x > 0)
      ans += x;
  }
  
  cout << ans << endl;
  return 0;
}
