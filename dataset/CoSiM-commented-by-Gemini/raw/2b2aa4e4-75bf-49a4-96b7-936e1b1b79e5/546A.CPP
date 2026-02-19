/**
 * @file 546A.CPP
 * @brief Solution to a computational problem involving arithmetic progression.
 * @details This program calculates the amount of money a person needs to borrow to buy a certain number of items with a linearly increasing cost.
 *
 * Algorithm:
 * The problem involves calculating the total cost of 'w' bananas. The first banana costs 'k', the second costs '2k', and so on, up to the 'w'-th banana which costs 'w*k'.
 * This forms an arithmetic progression. The total cost is the sum of this series: k * (1 + 2 + 3 + ... + w).
 * The sum of the first 'w' positive integers is given by the formula w * (w + 1) / 2.
 * Therefore, the total cost is k * w * (w + 1) / 2.
 * The program calculates this total cost and subtracts the initial amount of money 'n' the person has.
 * If the total cost is greater than 'n', the difference is the amount to be borrowed.
 * Otherwise, the person doesn't need to borrow any money, so the result is 0.
 *
 * Time Complexity: O(w) due to the loop iterating 'w' times. A more optimized solution could compute the sum in O(1) using the formula.
 * Space Complexity: O(1) as it uses a fixed amount of memory.
 */
#include <bits/stdc++.h>
using namespace std;

typedef long long int lli;

int main() {
  lli w, n, k, i, sum = 0;
  // k: cost of the first banana
  // n: initial amount of money
  // w: number of bananas to buy
  cin >> k >> n >> w;

  /**
   * Block Logic: This loop calculates the total cost of buying 'w' bananas.
   * Pre-condition: 'w' is the number of bananas to buy, 'k' is the base cost.
   * Invariant: At the end of each iteration 'i', 'sum' holds the total cost of buying 'i' bananas.
   */
  for (i = 1; i <= w; i++) {
    sum = sum + i * k;
  }

  /**
   * Block Logic: Determine the amount of money to borrow.
   * If the total 'sum' is greater than the initial money 'n', the amount to borrow is the difference.
   * Otherwise, no money needs to be borrowed.
   */
  if (sum > n)
    sum = sum - n;
  else
    sum = 0;

  // Output the final amount to be borrowed.
  cout << sum << endl;

  return 0;
}
