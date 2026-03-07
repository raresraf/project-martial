/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem.
 *
 * @details
 * This program calculates the minimum required capacity of a tram by tracking the
 * number of passengers over several stops. It determines the peak number of
 * passengers on the tram at any one time. This implementation stores the exit
 * and entry counts for all stops in arrays, which is not strictly necessary
 * for the solution.
 *
 * Algorithm:
 * 1. Read the number of stops, n.
 * 2. Initialize `sum` (current passengers) and `max` (peak passengers) to 0.
 * 3. Loop n times:
 *    a. Read the number of exiting passengers and store it. Subtract this from the `sum`.
 *    b. Read the number of entering passengers and store it. Add this to the `sum`.
 *    c. After updating the `sum`, compare it with `max`. If `sum` is greater,
 *       update `max`.
 * 4. Print `max` as the minimum required capacity.
 *
 * Time Complexity: O(N), where N is the number of stops.
 * Space Complexity: O(C), where C is the fixed size of the arrays (20000).
 *                   This is a constant but space-inefficient approach as the
 *                   problem can be solved with O(1) space.
 */
#include <iostream>

using namespace std;

int main() {
  int n, a[20000], b[20000], i, sum = 0, max = 0;
  cin >> n;
  /**
   * @brief Loop to simulate passenger flow across all stops.
   * This loop reads exit/entry counts, stores them, and updates the current
   * and maximum passenger counts simultaneously.
   */
  for (i = 0; i < n; i++) {
    // Read exiting passengers, store in array 'a', and update current passenger count.
    cin >> a[i];
    sum -= a[i];
    // Read entering passengers, store in array 'b', and update current passenger count.
    cin >> b[i];
    sum += b[i];
    // If the current passenger count is a new peak, update the max.
    if (max < sum) {
      max = sum;
    }
  }
  // The final 'max' value is the minimum required capacity.
  cout << max;
}
