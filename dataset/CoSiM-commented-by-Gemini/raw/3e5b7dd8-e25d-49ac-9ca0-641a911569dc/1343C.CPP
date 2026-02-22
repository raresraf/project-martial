// Problem Link : https://codeforces.com/problemset/problem/1343/C

/**
 * @file 1343C.CPP
 * @brief Solution for the "Alternating Subsequence" problem (Codeforces 1343C).
 *
 * @algorithm
 * The problem requires finding the maximum possible sum of an alternating subsequence.
 * An alternating subsequence is one where the signs of adjacent elements are
 * different. The goal is to maximize this sum by choosing the longest such subsequence.
 *
 * To achieve the maximum sum, we should pick the largest number from each
 * contiguous block of same-signed numbers.
 * For example, in `[1, 2, 3, -1, -2, 4, 5]`, the blocks are `[1, 2, 3]`,
 * `[-1, -2]`, and `[4, 5]`. We pick the largest from each: `3`, `-1`, and `5`.
 * The final sum is `3 + (-1) + 5 = 7`.
 *
 * This implementation finds these largest elements in a single pass:
 * 1. Iterate through the array, keeping track of the current block's maximum (`p`).
 * 2. As long as the elements have the same sign as `p`, update `p` if a larger
 *    element is found.
 * 3. When an element with an opposite sign is encountered, it signifies the end
 *    of the current block. The current maximum `p` is stored, and the new
 *    element becomes the maximum for the next block.
 * 4. After the loop, the maximum from the very last block is stored.
 * 5. Finally, sum up all the stored maximums.
 *
 * @complexity
 * - Time Complexity: O(N) for each test case, as it involves a single pass
 *   through the N elements.
 * - Space Complexity: O(N) to store the input array and O(N) in the worst case
 *   for the vector `b` (if the signs alternate for every element).
 */
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
/**
 * @brief Checks if two numbers have opposite signs using a bitwise XOR trick.
 * If signs are different, their most significant bits (sign bits) will differ.
 * The XOR operation will result in a number with the MSB set to 1, which is
 * negative when interpreted as a signed integer.
 * @return true if signs are opposite, false otherwise.
 */
bool oppositesigns(ll x, ll y) { return ((x ^ y) < 0); }
/**
 * @brief A simple comparison function. Can be replaced by `x > y`.
 * @return true if x is greater than y.
 */
bool large(ll x, ll y) {
  if (x > y)
    return 1;
  return 0;
}
int main() {
  // Fast I/O
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  ll t, n, p, i;
  cin >> t;
  // Loop for each test case.
  while (t--) {
    cin >> n;
    // Note: Using a Variable Length Array (VLA), a non-standard C++ extension.
    ll a[n];
    // `b` will store the maximum value from each alternating block.
    vector<ll> b;
    for (i = 0; i < n; i++)
      cin >> a[i];
    p = a[0]; // `p` tracks the maximum element of the current block.
    /**
     * @brief Main loop to find the maximum element in each block.
     */
    for (i = 0; i < n; i++) {
      // Check if the current element `a[i]` has the same sign as `p`.
      if (oppositesigns(p, a[i]) == 0) {
        // If signs are the same, we are in the same block.
        // Update `p` if `a[i]` is larger.
        if (large(a[i], p) == 1)
          p = a[i];
      } else {
        // If signs are different, the block has ended.
        // Add the maximum of the previous block to our list.
        b.push_back(p);
        // Start a new block with the current element.
        p = a[i];
      }
    }
    // Add the maximum from the very last block.
    b.push_back(p);
    ll sum = 0;
    // Calculate the sum of the maximums from all blocks.
    for (i = 0; i < b.size(); i++) {
      sum = sum + b[i];
    }
    cout << sum << endl;
  }
  return 0;
}
