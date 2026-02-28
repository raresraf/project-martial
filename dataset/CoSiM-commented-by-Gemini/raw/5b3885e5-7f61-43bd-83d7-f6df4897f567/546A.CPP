/**
 * @file 546A.CPP
 * @brief Solution for the "Soldier and Bananas" problem.
 * @details This program calculates the amount of money a soldier needs to borrow.
 * The cost of the first banana is `k`, the second is `2k`, and so on, up to the `w`-th banana costing `w*k`.
 * The soldier has `n` dollars initially. The program computes the total cost and determines
 * how much money the soldier must borrow.
 *
 * @algorithm Arithmetic Series Summation
 * The total cost of `w` bananas forms an arithmetic series: k + 2k + ... + wk = k * (1 + 2 + ... + w).
 * The sum of the first `w` positive integers is given by the formula w * (w + 1) / 2.
 * Therefore, the total cost is `k * w * (w + 1) / 2`.
 * The amount to borrow is the total cost minus the initial money `n`. If the result is
 * negative or zero, the soldier doesn't need to borrow, so the answer is 0.
 *
 * @complexity
 * - Time: O(1), as the solution is a direct calculation using a mathematical formula.
 * - Space: O(1), for storing a few integer variables.
 */

#include <cstdio> // For scanf and printf

// Using the standard namespace to avoid prefixing std:: to standard library elements.
using namespace std;

int main() {
  // k: cost of the first banana
  // n: initial amount of money the soldier has
  // w: number of bananas he wants to buy
  int k, n, w;
  
  // The total amount of money the soldier needs to borrow.
  int amount;
  
  // Read the input values from standard input.
  scanf("%d %d %d", &k, &n, &w);
  
  /**
   * @brief Calculate the total cost and the amount to borrow.
   * The total cost is the sum of an arithmetic progression: k * (1 + 2 + ... + w).
   * This is calculated using the formula for the sum of the first w integers: w * (w + 1) / 2.
   * `amount` is the difference between the total cost and the soldier's initial money.
   */
  amount = (w * (w + 1) * k / 2) - n;
  
  /**
   * @brief Print the result.
   * If the calculated `amount` is positive, it means the soldier needs to borrow that much.
   * If `amount` is zero or negative, the soldier has enough money, so he needs to borrow 0.
   * The ternary operator `(amount > 0 ? amount : 0)` handles this logic concisely.
   */
  printf("%d", amount > 0 ? amount : 0);

  return 0; // Indicate successful execution.
}
