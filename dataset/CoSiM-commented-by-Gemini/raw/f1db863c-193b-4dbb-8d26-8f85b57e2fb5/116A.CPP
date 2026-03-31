/**
 * @file 116A.CPP
 * @brief Calculates the minimum required capacity of a tram.
 *
 * This program solves the "Tram" problem. It simulates the passenger flow in a tram
 * across several stops to determine the maximum number of passengers on board at any
 * given time. This maximum number represents the minimum capacity the tram must have.
 * It reads the number of stops `n`, and for each stop, it reads the number of
 * passengers exiting and entering.
 *
 * @author f1db863c-193b-4dbb-8d26-8f85b57e2fb5
 *
 * @algorithm Simulation and Tracking Maximum Value
 * @complexity
 * Time: O(N), where N is the number of stops, due to a single loop.
 * Space: O(1) for the essential variables. The declared array `a` is unused.
 */
#include <algorithm>
#include <bits/stdc++.h>
#include <cmath>
#include <iostream>
using namespace std;
// The array `a` is declared globally but is not used in the main logic.
int a[100010];
// The array `b` is commented out and thus not part of the program.
// int b[100010];
int main() {
  // `n` is the number of tram stops.
  int n;
  cin >> n;
  // `a` and `b` will store the number of exiting and entering passengers at each stop.
  int a, b;
  // `ma` (max) will store the maximum number of passengers on the tram at any point.
  // `sum` will track the current number of passengers on the tram.
  long int ma = 0, sum = 0;

  /**
   * @brief Loop to process each of the `n` stops.
   * Pre-condition: `sum` holds the passenger count from before the current stop,
   *              and `ma` holds the maximum count seen so far.
   * Invariant: At the end of each iteration, `sum` is updated with the new
   *            passenger count, and `ma` is updated if `sum` sets a new maximum.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    // Update the current passenger count: add entering, subtract exiting.
    sum = b - a + sum;
    /**
     * @brief Check if the current passenger count is the new maximum.
     * If the number of passengers after the current stop is greater than the
     * maximum recorded so far, update the maximum.
     */
    if (sum > ma)
      ma = sum;
  }
  // Output the minimum required capacity, which is the maximum passenger count found.
  cout << ma;
  return 0;
}