/**
 * @file 116A.CPP
 * @brief Solution for a competitive programming problem calculating tram capacity.
 * @details This program calculates the minimum required capacity of a tram based
 * on the number of passengers entering and exiting at each stop. It simulates
 * the change in passenger count and tracks the maximum number of passengers
 * inside the tram at any given time.
 *
 * Algorithm:
 * - Read the total number of stops.
 * - Iterate through each stop, reading the number of passengers exiting and entering.
 * - Maintain a running sum of the current number of passengers in the tram.
 * - After each stop, update a 'max' variable if the current passenger count is
 *   the highest yet seen.
 * - The final 'max' value represents the minimum capacity the tram must have.
 *
 * Time Complexity: O(N), where N is the number of stops.
 * Space Complexity: O(1), as only a few variables are used regardless of input size.
 */
#include <iostream>
using namespace std;

int main() {
  int stn;
  // Read the number of tram stops.
  cin >> stn;
  int a, b, sum = 0, max = 0;

  // Invariant: 'sum' holds the number of passengers currently on the tram.
  // 'max' holds the maximum value 'sum' has reached so far.
  while (stn--) {
    // Read the number of passengers exiting (a) and entering (b).
    cin >> a >> b;
    // Update the current passenger count.
    sum += (b - a);
    // If the current passenger count exceeds the max capacity seen so far,
    // update the max capacity.
    if (max < sum) {
      max = sum;
    }
  }
  // Output the minimum required capacity.
  cout << max;
  return 0;
}