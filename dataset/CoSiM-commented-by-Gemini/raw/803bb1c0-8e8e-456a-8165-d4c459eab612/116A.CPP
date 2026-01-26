/**
 * @file 116A.CPP
 * @brief Solution to a competitive programming problem about calculating tram capacity.
 * @details This program calculates the minimum required capacity of a tram. It simulates
 * 'n' stops, where at each stop, a certain number of passengers exit and enter.
 * The goal is to find the maximum number of passengers on the tram at any point in time,
 * which determines the minimum capacity needed.
 *
 * @algorithm Simulation and tracking maximum value.
 * @complexity
 * Time complexity: O(n), where 'n' is the number of tram stops, as it iterates through each stop once.
 * Space complexity: O(1), as it uses a constant amount of extra space.
 */
#include <bits/stdc++.h>
using namespace std;
/**
 * @brief Main function to execute the solution.
 */
int main() {
  /**
   * @param n The number of tram stops.
   * @param a The number of passengers exiting at a stop.
   * @param b The number of passengers entering at a stop.
   * @param k The maximum number of passengers on the tram at any time (the result).
   * @param res The current number of passengers on the tram.
   */
  int n, a, b, k = 0, res = 0;
  cin >> n;
  /**
   * @brief This loop simulates the tram's journey through 'n' stops.
   * Pre-condition: 'n' contains the total number of stops.
   * Invariant: After each iteration, 'res' holds the current number of passengers,
   * and 'k' holds the maximum number of passengers seen so far.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    res -= a;
    res += b;
    if (res > k) {
      k = res;
    }
  }
  cout << k << endl;
}
