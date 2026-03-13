/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem from Codeforces.
 *
 * This program calculates the amount of money a soldier needs to borrow to
 * buy a certain number of bananas. The cost of the first banana is k, the
 * second is 2k, and so on. The soldier has n dollars and wants to buy w
 * bananas.
 *
 * @see https://codeforces.com/problemset/problem/546/A
 */
#include <stdio.h>
int main() {
  /**
   * @brief Input variables.
   *
   * @param k The cost of the first banana.
   * @param n The initial amount of money the soldier has.
   * @param w The number of bananas the soldier wants to buy.
   * @param s The total cost of the bananas.
   * @param ans The amount of money the soldier needs to borrow.
   */
  long long n, w, k, s, ans;
  scanf("%lld %lld %lld", &k, &n, &w);
  /**
   * @brief Calculates the total cost of the bananas.
   *
   * The total cost is the sum of an arithmetic progression:
   * k + 2k + 3k + ... + wk = k * (1 + 2 + 3 + ... + w)
   * The sum of the first w integers is w * (w + 1) / 2.
   */
  s = k * (w * (w + 1) / 2);
  ans = s - n;
  /**
   * @brief Determines the amount to borrow.
   *
   * If the total cost `s` is greater than the money the soldier has `n`,
   * the difference is the amount to borrow. Otherwise, the soldier doesn't
   * need to borrow anything.
   */
  if (ans > 0) {
    printf("%lld", ans);
  } else
    printf("0");
  return 0;
}