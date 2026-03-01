/**
 * @file 1343C.CPP
 * @brief Solution for the "Alternating Subsequence" problem.
 *
 * @problem Given an array of non-zero integers, find an alternating subsequence
 * (where signs of adjacent elements differ) that has the maximum possible length,
 * and among all such subsequences, find the one with the maximum sum.
 *
 * @algorithm This problem can be solved with a greedy approach. An alternating
 * subsequence of maximum length is formed by picking exactly one element from each
 * contiguous block of same-signed numbers. To maximize the sum, we must pick the
 * largest element from each of these blocks.
 *
 * The algorithm is as follows:
 * 1. Iterate through the array, identifying contiguous blocks of numbers with the
 *    same sign.
 * 2. For each block, find its maximum element.
 * 3. Sum up these maximum elements from all blocks to get the final answer.
 *
 * @time_complexity O(N) for each test case. The two pointers, `i` and `j`,
 * together traverse the array exactly once.
 * @space_complexity O(N) to store the input array.
 */
#include <bits/stdc++.h>
#include <iostream>
using namespace std;
#define ll long long
#define sd long long

/**
 * @brief Checks if two integers have the same sign.
 * @param x The first integer.
 * @param y The second integer.
 * @return True if both are positive or both are negative, false otherwise.
 */
bool find_(int x, int y) {
  if ((x > 0 && y > 0) || (x < 0 && y < 0))
    return true;
  else
    return false;
}
int main() {
  int t;
  cin >> t;
  while (t--) {
    int n;
    cin >> n;
    // Note: Using a variable-length array (VLA), which is not standard C++.
    int a[n + 1];
    for (int i = 0; i < n; i++)
      cin >> a[i];

    int i = 0, j = 0;
    ll sum = 0;
    // Outer loop iterates through the array, block by block.
    while (i < n) {
      j = i;
      // `num` will store the maximum element in the current block.
      int num = a[j];
      // Inner loop finds the end of the current block of same-signed numbers.
      while (j + 1 < n && find_(a[j], a[j + 1])) {
        // Update the maximum element found in this block.
        num = max(num, a[j + 1]);
        j++;
      }
      // Add the maximum element of the completed block to the total sum.
      sum += num;
      // Move the start pointer to the beginning of the next block.
      i = j + 1;
    }
    cout << sum << endl;
  }
  return 0;
}
