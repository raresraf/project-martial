/**
 * @file 116A.CPP
 * @brief An incorrect attempt to solve the "Tram" problem.
 *
 * @details
 * This program is intended to calculate the minimum required capacity of a tram by
 * finding the peak number of passengers. However, the implementation contains
 * several critical flaws, including out-of-bounds array access and incorrect logic
 * for calculating the passenger count, and will not produce the correct answer.
 *
 * Flaws in the Algorithm:
 * 1.  **Buffer Overflow**: The code uses 1-based indexing (`i` from 1 to `n`) on
 *     0-indexed C++ arrays of size `n`, causing it to write past the allocated
 *     memory when `i` equals `n`.
 * 2.  **Incorrect Logic**: The core loop for calculating the number of passengers
 *     at each stop is logically incorrect. It does not maintain a proper running
 *     total of the current passengers. It incorrectly mixes values from
 *     different stops and overwrites its own input data.
 *
 * Intended (but flawed) approach:
 * 1. Read all exit and entry data into arrays `a` and `b`.
 * 2. Attempt to calculate the passenger count at each stop and store it in `tmp`.
 * 3. Find the maximum value in the `tmp` array.
 *
 * Time Complexity: O(N), due to the loops.
 * Space Complexity: O(N), for the three variable-length arrays.
 */
#include <iostream>

using namespace std;

int main() {

  int n;
  int ans = 0;

  cin >> n;

  // VLAs are a non-standard C++ extension.
  int a[n] = {0};
  int b[n] = {0};
  int tmp[n] = {0};

  /**
   * @brief Reads all exit and entry data for all stops.
   * CRITICAL FLAW: This loop runs from i=1 to n, but the arrays `a` and `b` are
   * 0-indexed (0 to n-1). When i=n, this causes a buffer overflow.
   */
  for (int i = 1; i <= n; ++i) {
    cin >> a[i] >> b[i];
  }

  // Base case: passengers after the first stop.
  tmp[0] = b[1];

  /**
   * @brief Attempts to calculate the number of passengers at subsequent stops.
   * CRITICAL FLAW: The logic here is incorrect for tracking the running total
   * of passengers on the tram. It also overwrites the input array `b`.
   */
  for (int i = 2; i <= n; ++i) {

    tmp[i - 1] = b[i - 1] - a[i];
    tmp[i - 1] += b[i];

    b[i] = tmp[i - 1];
  }

  /**
   * @brief Finds the maximum value in the incorrectly computed `tmp` array.
   */
  for (int i = 0; i < n; ++i) {
    if (tmp[i] > ans) {
      ans = tmp[i];
    }
    // cout<<tmp[i];
  }

  cout << ans;

  return 0;
}
