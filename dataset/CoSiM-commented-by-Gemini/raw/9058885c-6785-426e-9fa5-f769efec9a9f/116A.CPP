/**
 * @file 116A.CPP
 * @brief Solution for the "Tram" capacity problem (e.g., Codeforces 116A).
 * @details This program simulates passenger flow on a tram across several stops to
 * find the maximum number of passengers on board at any given time, which determines
 * the minimum required capacity.
 *
 * @note This implementation uses several redundant variables, making it less clear
 * than it could be. The logic could be simplified.
 *
 * @algorithm Iterative Simulation
 * @complexity
 * - Time: O(n), where 'n' is the number of stops, due to a single loop.
 * - Space: O(1).
 */
#include <iostream>

using namespace std;

int main() {
  // --- Variable Declarations ---
  // n: the number of stops.
  // a: passengers exiting.
  // b: passengers entering.
  // peopleIn, peopleOut: Redundant copies of a and b.
  // mAx: Stores the net change for the current stop (b - a).
  // total: The current number of passengers on the tram.
  // res: The result, tracking the maximum `total` seen so far.
  int n = 0, a = 0, b = 0, peopleIn = 0, peopleOut = 0, mAx = 0, total = 0,
      res = 0;
  
  cin >> n;

  /**
   * @block Main Simulation Loop
   * @logic Iterates through each stop, updating the current passenger count
   * and tracking the maximum count.
   */
  for (int i = 0; i < n; i++) {
    // Read the number of passengers who leave (a) and enter (b).
    cin >> a;
    cin >> b;

    // These assignments are redundant; `a` and `b` could be used directly.
    peopleOut = a;
    peopleIn = b;

    // Calculate the net change in passengers for this stop.
    mAx = peopleIn - peopleOut;
    // Update the total number of passengers on the tram.
    total = total + mAx;

    // If the new total is the largest seen so far, update the result.
    if (total > res) {
      res = total;
    }
  }

  // Print the required minimum capacity.
  cout << res;

  return 0;
}
