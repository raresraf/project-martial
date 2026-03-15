/**
 * @file 1343C.CPP
 * @brief Solution to the "Alternating Subsequence" problem.
 * @details The problem asks to find the maximum sum of an alternating subsequence of maximum length.
 * An alternating subsequence is one where the signs of adjacent elements differ.
 *
 * @algorithm
 * The core insight is a greedy approach. To maximize the sum while maintaining the longest possible
 * alternating sequence, we should pick exactly one number from each contiguous block of same-signed numbers.
 * To maximize the sum, this one number must be the largest within that block.
 *
 * The implementation works as follows:
 * 1. Iterate through the array block by block, where a block is a contiguous sequence of numbers with the same sign.
 * 2. For each block, find the maximum element.
 * 3. Add this maximum element to a running total.
 * 4. Advance the main iterator to the end of the current block to start the next iteration at the beginning of the next block.
 *
 * @complexity
 * - Time: O(N) for each test case, where N is the number of elements in the input array. Each element is visited a constant number of times.
 * - Space: O(N) to store the input vector for each test case.
 */
#include <bits/stdc++.h>
using namespace std;

typedef long long int ll;

int main() {
  // #ifndef ONLINE_JUDGE
  // freopen("input.txt","r",stdin);
  // freopen("output.txt","w",stdout);
  // #endif

  // A lambda function to determine the sign of an integer (-1 for negative, 1 for positive).
  auto sgn = [&](int x) {
    if (x > 0) {
      return 1;
    } else {
      return -1;
    }
  };

  int t; // Number of test cases.
  cin >> t;
  while (t--) {
    int n;
    ll total = 0; // The final maximum sum.
    cin >> n;
    std::vector<int> v(n);
    for (auto &it : v)
      cin >> it;

    /**
     * @brief Main loop to iterate through the array, processing one block of same-signed numbers at a time.
     */
    for (int i = 0; i < n; ++i) {
      int cur = v[i]; // Stores the maximum element found so far in the current block.
      int j = i;      // Inner loop index to find the end of the block.

      /**
       * @brief Inner loop to find the end of the current block of same-signed numbers.
       * Invariant: `cur` holds the maximum value in `v[i...j-1]`.
       */
      while (j < n && sgn(v[i]) == sgn(v[j])) {
        cur = max(cur, v[j]);
        ++j;
      }
      
      // Add the largest element from the processed block to the total sum.
      total += cur;
      
      // Advance the outer loop index `i` to the end of the block we just processed.
      // The `++i` in the for loop will then move it to the start of the next block.
      i = j - 1;
    }
    cout << total << "\n";
  }

  return 0;
}
