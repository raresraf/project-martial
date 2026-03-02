#include <iostream>
using namespace std;

/**
 * @file 116A.CPP
 * @brief Solution to the "Tram" problem from Codeforces (Problem 116A).
 * @details This program calculates the minimum required capacity of a tram by
 * tracking the maximum number of passengers on board at any given time.
 *
 * Algorithm:
 * The program simulates the tram's journey stop by stop. It maintains two key variables:
 * one for the current number of passengers and another for the maximum number of passengers
 * observed so far.
 * 1. It reads the total number of stops, `n`.
 * 2. It loops `n` times, once for each stop.
 * 3. In each iteration, it reads the number of passengers exiting and entering.
 * 4. The current passenger count is updated by subtracting the exiting passengers and
 *    adding the entering ones.
 * 5. This new current count is compared with the maximum count seen so far. If the
 *    current count is larger, the maximum is updated.
 * 6. After the loop, the final maximum value is the minimum capacity the tram needs.
 *
 * Time Complexity: O(n), where `n` is the number of stops, due to the single loop.
 * Space Complexity: O(1).
 */
int main() {
  int n, i, a, b, c = 0, m = 0;
  cin >> n;
  
  // c: current number of passengers on the tram.
  // m: maximum number of passengers observed so far (minimum capacity).
  
  /**
   * @brief Loop through each of the `n` stops.
   */
  for (i = 1; i <= n; i++) {
    // Read passengers exiting (a) and entering (b).
    cin >> a >> b;
    // Update the current number of passengers.
    c = c - a + b;
    // If the current number of passengers is the highest yet, update the maximum.
    if (c > m)
      m = c;
  }
  // Print the minimum required capacity.
  cout << m;
  return 0;
}
