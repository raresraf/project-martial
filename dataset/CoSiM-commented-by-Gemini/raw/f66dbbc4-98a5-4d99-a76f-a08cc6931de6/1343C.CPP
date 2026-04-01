/**
 * @file 1343C.CPP
 * @brief Solves the "Alternating Subsequence" problem from Codeforces (1343C).
 * @details The goal is to find the maximum sum of an alternating subsequence.
 * An alternating subsequence is one where the signs of consecutive elements differ.
 * The problem requires selecting exactly one element from each contiguous block of
 * same-signed numbers to maximize the total sum. The optimal strategy is to pick
 * the largest number from each block.
 *
 * Algorithm: Single-pass Greedy Approach.
 * 1. Iterate through the array, processing elements in blocks of the same sign.
 * 2. For each block, find the maximum element.
 * 3. Add this maximum element to the total sum.
 * 4. Move to the next block (which will have a different sign) and repeat.
 * This implementation achieves this by iterating through the array and keeping track
 * of the current block's maximum.
 *
 * Time Complexity: O(N) for each test case, as we iterate through the array once.
 * Space Complexity: O(N) to store the input array for each test case.
 */
// https://codeforces.com/contest/1343/problem/C

#include <iostream>

#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Determines the sign of an integer.
 * @param n The integer to check.
 * @return 1 if n is positive, -1 if n is negative, and 0 if n is zero.
 */
int sign(int n)

{

  if (n == 0)
    return 0;

  if (n > 0)
    return 1;

  return -1;
}

int main()

{

  int t;

  cin >> t;

  /**
   * @brief Loop to handle multiple test cases.
   */
  while (t--)

  {

    int n;

    cin >> n;

    int a[n];

    int sign_prev = 0;

    int prev_val = 0;

    int temp;

    long long int ans = 0;

    /**
     * @brief Iterates through the array to build the max-sum alternating subsequence.
     * This loop identifies blocks of same-signed numbers and incorporates the
     * largest value from each block into the final sum `ans`.
     */
    for (int i = 0; i < n; i++)

    {

      cin >> a[i];

      if (i == 0)

      {
        // Initialize with the first element.
        ans += a[i];

        sign_prev = sign(a[i]);

        prev_val = a[i];

      }

      else

      {

        temp = sign(a[i]);

        // If the current element is in the same sign block.
        if (temp == sign_prev)

        {
          // Update the sum by replacing the previous max of this block
          // with the new max if the current element is larger.
          ans = ans - prev_val + max(prev_val, a[i]);

          prev_val = max(prev_val, a[i]);

        }

        else

        {
          // If the current element starts a new sign block, add it to the sum.
          ans += a[i];

          prev_val = a[i];

          sign_prev = temp;
        }
      }
    }

    cout << ans << endl;
  }
}