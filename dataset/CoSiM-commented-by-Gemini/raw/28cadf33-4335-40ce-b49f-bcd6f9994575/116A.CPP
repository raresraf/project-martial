/**
 * @file 116A.CPP
 * @brief Solution to the "Tram" problem (116A) from Codeforces.
 * @details This program calculates the minimum required capacity of a tram by finding the
 * maximum number of passengers on board at any single point in time.
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm This solution simulates the passenger flow stop by stop.
 * 1. Read the total number of stops (n).
 * 2. Initialize `ans` (current passengers), `cmp` (max capacity), and `f` (final answer) to 0.
 * 3. Loop 'n' times for each stop.
 * 4. In each iteration, read the exiting and entering passenger counts. Although they are
 *    stored in arrays `in` and `out`, this is not necessary as they are used immediately.
 * 5. Update the current passenger count: `ans = ans - out[i] + in[i]`.
 * 6. If the current count `ans` is greater than the max capacity seen so far (`cmp`),
 *    update both `cmp` and `f` to this new maximum.
 * 7. After the loop, `f` holds the answer.
 *
 * @note This implementation is correct but has some inefficiencies. It uses O(N) space
 * for arrays that are not needed, and it uses two redundant variables (`cmp` and `f`)
 * to track the maximum value.
 *
 * @complexity
 * Time Complexity: O(N), for the single loop through the stops.
 * Space Complexity: O(N), due to the `in` and `out` arrays used to store passenger data for all stops.
 */
#include <bits/stdc++.h>
using namespace std;

// Arrays to store all entering and exiting passenger counts. This is not
// strictly necessary as the values can be processed one stop at a time.
int in[1005];
int out[1005];

int main() {
  // `cmp` and `f` are used redundantly to track the same maximum value.
  // `ans` is used to track the current passenger count.
  int n, cmp = 0, ans = 0, f = 0;
  scanf("%d", &n);
  int i;
  /**
   * @block
   * @brief Simulates passenger flow over 'n' stops to find the peak occupancy.
   * @invariant `ans` holds the current passenger count. `cmp` and `f` hold the max count seen so far.
   * @pre 'n' is the number of stops to simulate.
   * @post 'f' contains the minimum required tram capacity.
   */
  for (i = 0; i < n; i++) {
    scanf("%d %d", &out[i], &in[i]);
    // Update current passenger count.
    ans -= out[i];
    ans += in[i];
    // If the current passenger count is a new maximum, update the max trackers.
    if (cmp < ans) {
      f = ans;
      cmp = ans;
    }
  }
  printf("%d\n", f);
  return 0;
}
