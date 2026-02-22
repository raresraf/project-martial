/**
 * @file 1343C.CPP
 * @brief Solution to Codeforces problem 1343C - Alternating Subsequence.
 *
 * @algorithm
 * The problem requires finding the maximum possible sum of an alternating subsequence
 * of maximum length. An alternating subsequence is one where the signs of any two
 * adjacent elements are different.
 *
 * This can be solved by partitioning the input array into contiguous blocks of
 * elements with the same sign. From each of these blocks, we must choose the
 * largest element. The final answer is the sum of these largest elements.
 *
 * This solution implements a greedy, single-pass algorithm:
 * 1. Read the number of test cases `t`.
 * 2. For each test case, initialize `sum = 0`.
 * 3. Read the first element into a variable `p`, which will track the maximum
 *    element of the current same-sign block.
 * 4. Iterate through the rest of the array, reading one element `c` at a time.
 * 5. Check if `p` and `c` have the same sign using the condition `p * c > 0`.
 *    - If they have the same sign, we are in the same block. Update `p` to be
 *      the maximum of the current `p` and `c`.
 *    - If they have different signs, the previous block has ended. Add its
 *      maximum element (`p`) to the total `sum`, and start a new block by
 *      setting `p = c`.
 * 6. After the loop, the maximum element of the very last block is still in `p`.
 *    Add it to the sum and print the result.
 *
 * @complexity
 * - Time Complexity: O(N) for each test case, as it involves a single pass
 *   through the array.
 * - Space Complexity: O(1), as it processes the array elements one by one
 *   without storing the entire array.
 */
#include <iostream>
using namespace std;
#define ll long long

int main() {
  ll t, n, p, c, sum;

  cin >> t;

  while (t--) {
    sum = 0;
    cin >> n;
    // Read the first element to initialize the first block.
    cin >> p;
    n--;
    /**
     * @brief Process the rest of the array element by element.
     */
    while (n--) {
      cin >> c;
      /**
       * @brief A positive product means `p` and `c` have the same sign.
       * This is the condition for being in the same contiguous block.
       */
      if (p * c > 0) { // check is it in same sign or not
        // If they have the same sign, keep track of the maximum element.
        if (c > p)
          p = c;
      } else {
        // If signs differ, the previous block has ended.
        // Add the maximum element from that block to the sum.
        sum += p; // add the max from same sign elements
        // Start a new block with the current element `c`.
        p = c;
      }
    }
    // Add the maximum element from the very last block and print the total sum.
    cout << sum + p << endl; // add max element from last sub-array of same sign
  }

  return 0;
}