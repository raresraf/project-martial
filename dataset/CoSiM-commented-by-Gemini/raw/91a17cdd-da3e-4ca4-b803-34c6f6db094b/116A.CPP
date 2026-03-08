/**
 * @file 116A.CPP
 * @brief Solution for the "Tram" capacity simulation problem (e.g., Codeforces 116A).
 * @details This program calculates the minimum required capacity for a tram by simulating
 * the change in passenger numbers over a series of stops and finding the peak occupancy.
 *
 * @algorithm Iterative Simulation
 * @complexity
 * - Time: O(n), where 'n' is the number of tram stops, as it involves a single loop.
 * - Space: O(1), as it uses a fixed number of variables.
 */
#include <bits/stdc++.h>

using namespace std;

#define ll long long // Unused macro

int main() {
  // n: the number of stops.
  int n;
  cin >> n;

  // c: the current number of passengers on the tram.
  // m: the maximum number of passengers observed so far.
  int c = 0;
  int m = 0;

  // a: passengers exiting.
  // b: passengers entering.
  int a, b;

  /**
   * @block Main Simulation Loop
   * @logic Iterates through each stop, updating the current passenger count and
   * tracking the maximum value.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    // Update the current passenger count.
    c -= a;
    c += b;
    // If the current count is greater than the max seen, update the max.
    if (c > m)
      m = c;
  }
  
  // Output the required capacity.
  cout << m << "
";
  
  return 0;
}
