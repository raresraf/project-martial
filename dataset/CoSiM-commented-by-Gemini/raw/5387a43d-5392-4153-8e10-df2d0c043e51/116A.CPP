/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem from Codeforces.
 * @details The problem is to calculate the minimum required capacity of a tram.
 * This is determined by finding the maximum number of passengers inside the
 * tram at any single point in time during its journey.
 *
 * @algorithm
 * The solution simulates the tram's journey stop by stop, keeping track of the
 * current number of passengers and the maximum number seen so far.
 * 1. Read the number of stops, n.
 * 2. Initialize `sum` to 0 (to track current passengers) and `ans` to 0 (to
 *    track the maximum passengers seen).
 * 3. Loop n times for each stop:
 *    a. Read the number of exiting passengers (`x`) and entering passengers (`y`).
 *    b. Update the current passenger count: `sum = sum - x + y`.
 *    c. Update the maximum capacity required (`ans`) by taking the maximum of
 *       the current `ans` and the current passenger count `sum`.
 * 4. After the loop, `ans` will hold the peak passenger count, which is the
 *    minimum capacity needed. Print `ans`.
 *
 * @complexity
 * Time Complexity: O(n), where n is the number of stops, due to a single loop.
 * Space Complexity: O(1).
 */
#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  int n; // The number of stops.
  int x, y; // x: passengers exiting, y: passengers entering.
  int ans = 0; // Stores the maximum number of passengers seen (the answer).
  int sum = 0; // Stores the current number of passengers on the tram.
  
  cin >> n;
  
  /**
   * @block
   * @brief Simulate the passenger count at each stop.
   */
  for (int i = 0; i < n; i++) {
    cin >> x >> y;
    // Update the current passenger count.
    sum -= x;
    sum += y;
    // Update the maximum capacity required so far.
    ans = max(ans, sum);
  }
  
  cout << ans << endl;
}
