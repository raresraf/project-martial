/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem.
 * @details A soldier wants to buy `w` bananas. The cost of the bananas increases linearly:
 * the first costs `k`, the second `2k`, ..., the `w`-th costs `w*k`. The soldier has `n` dollars.
 * This program calculates the total cost of all `w` bananas and determines how much money
 * the soldier needs to borrow. If the soldier has enough money, the amount to borrow is 0.
 *
 * @algorithm Iterative summation of an arithmetic series.
 * @complexity
 * Time complexity: O(W), where W is the number of bananas, due to the loop to calculate the total cost.
 * An O(1) solution is possible by using the formula for the sum of an arithmetic series: cost = k * W * (W+1) / 2.
 * Space complexity: O(1).
 */
#include <stdio.h>
int main() {
  // k: the cost of the first banana.
  // n: the initial amount of money the soldier has.
  // w: the number of bananas the soldier wants to buy.
  // i: loop counter.
  // s: the total cost of all bananas.
  // b: the amount of money to borrow.
  // c: the cost of the current banana.
  int k, n, w, i, s = 0, b, c = 1;
  scanf("%d %d %d", &k, &n, &w);
  
  /**
   * @brief Calculates the total cost of `w` bananas.
   * @invariant In each iteration `i`, adds the cost of the i-th banana (`k*i`) to the total sum `s`.
   */
  for (i = 1; i <= w; i++) {
    c = k * i;
    s = s + c;
  }
  
  // Calculate the amount of money the soldier needs to borrow.
  b = s - n;
  
  /**
   * @brief Determine the output.
   * @post If the total cost `s` is less than or equal to the soldier's money `n`, print 0.
   * Otherwise, print the calculated amount to borrow `b`.
   */
  if (s <= n)
    printf("0");
  else
    printf("%d\n", b);
    
  return 0;
}
