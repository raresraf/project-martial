/**
 * @file 116A.CPP
 * @brief Calculates the minimum capacity of a tram.
 *
 * This program solves the "Tram" problem from Codeforces. It simulates the
 * passenger flow on a tram over several stops to find the peak occupancy,
 * which determines the minimum required capacity of the tram.
 *
 * @algorithm Simulation / Maximum Tracking
 * @complexity
 * - Time: O(N), where N is the number of stops, because of the single loop
 *   iterating through the stops.
 * - Space: O(1), as a fixed amount of memory is used regardless of the input size.
 */
#include <iostream>
using namespace std;
int main() {
  // i: loop counter.
  // max: tracks the maximum number of passengers on the tram at any one time.
  // a: passengers exiting, b: passengers entering.
  // curr: tracks the current number of passengers on the tram.
  // n: the total number of stops.
  int i, max(0), a, b, curr(0), n;
  cin >> n;

  /**
   * @brief Block Logic: Iterates through each stop to simulate passenger changes.
   * Invariant: At the end of each iteration, `curr` reflects the current passenger
   *            count, and `max` holds the highest value `curr` has reached so far.
   */
  for (i = 0; i < n; i++) {
    cin >> a >> b;
    // Update the current passenger count by subtracting those who exit and adding those who enter.
    curr += -a + b;
    // If the current occupancy is greater than the max recorded, update the max.
    if (curr > max)
      max = curr;
  }
  
  // Print the maximum capacity required.
  cout << max << endl;
  return 0;
}