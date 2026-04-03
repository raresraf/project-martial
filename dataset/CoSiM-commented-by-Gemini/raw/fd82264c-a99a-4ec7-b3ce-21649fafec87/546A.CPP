/**
 * @file 546A.cpp
 * @brief Solution to the "Soldier and Bananas" problem.
 * A soldier wants to buy `w` bananas. The first banana costs `k`, the second
 * `2k`, and so on, up to the `w`-th banana costing `w*k`. The soldier has `n`
 * dollars. This program calculates the total cost and determines how much
 * money the soldier needs to borrow.
 *
 * Algorithm:
 * 1. Read the cost of the first banana `k`, the soldier's initial money `n`,
 *    and the number of bananas to buy `w`.
 * 2. Calculate the total cost of `w` bananas. This is the sum of an arithmetic
 *    series: k * (1 + 2 + ... + w) = k * w * (w + 1) / 2.
 *    The code calculates this iteratively.
 * 3. Calculate the amount to borrow: `total_cost - n`.
 * 4. If the amount to borrow is positive, print it. Otherwise, print 0.
 *
 * Time Complexity: O(W), where W is the number of bananas.
 * Space Complexity: O(1).
 */
#include <stdio.h>
int main() {
  // k: cost of the first banana, n: initial money, w: number of bananas.
  long long int k, n, w, i, sum = 0;
  scanf("%lld %lld %lld", &k, &n, &w);

  // Calculate the total cost of w bananas.
  for (i = 1; i <= w; i++) {
    sum += (i * k);
  }

  // If the total cost is greater than the initial money, print the difference.
  if (sum > n)
    printf("%lld", sum - n);
  else // Otherwise, the soldier doesn't need to borrow anything.
    printf("0");

  return 0;
}