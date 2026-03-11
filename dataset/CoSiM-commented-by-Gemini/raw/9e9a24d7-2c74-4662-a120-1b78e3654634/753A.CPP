/**
 * @file 753A.CPP
 * @brief Solves a number partitioning problem using a greedy approach.
 *
 * This program takes an integer 'n' and partitions it into the maximum
 * possible number of distinct positive integers. The sum of these integers
 * must equal 'n'.
 *
 * Algorithm: Greedy algorithm. Sequentially take the smallest distinct positive
 * integers (1, 2, 3, ...) until the remaining value of 'n' is less than or
 * equal to the next integer in the sequence. The remainder is then added to
 * the last integer taken to ensure the sum is exactly 'n'.
 *
 * Time Complexity: O(sqrt(n)), as the loop runs approximately sqrt(n) times
 *                  because the sum of the first k integers is k*(k+1)/2.
 * Space Complexity: O(sqrt(n)) to store the resulting partition.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n; // The integer to be partitioned.
  int k; // Not used for input, but to track the number of partitions.
  int i; // Loop counter and the current integer to be used in the partition.
  
  cin >> n;
  
  // dp array will store the partition members.
  // The size is 'n' which is an overestimation but safe.
  int dp[n] = {};
  i = 1;

  /**
   * @brief Greedily find the sequence of distinct integers.
   *
   * Pre-condition: 'n' is the integer to partition.
   * Invariant: At each step, we subtract the smallest available distinct integer 'i'
   *            from 'n' and add 'i' to our set of partitions.
   */
  while (i <= n) {
    dp[i] = i;
    n -= i;
    i++;
  }
  
  /**
   * @brief Distribute the remainder.
   *
   * Logic: If there is any remainder left in 'n', it is added to the last
   *        and largest number in our partition to maintain the distinctness
   *        of the numbers while ensuring their sum equals the original 'n'.
   */
  if (n) {
    dp[i - 1] += n;
  }
  
  k = i;
  
  // Output the number of integers in the partition.
  cout << k - 1 << endl;
  
  // Output the integers in the partition.
  for (i = 1; i < k; i++)
    cout << dp[i] << " ";
}
