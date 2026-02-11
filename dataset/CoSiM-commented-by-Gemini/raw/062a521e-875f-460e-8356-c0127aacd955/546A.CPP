/**
 * @file 546A.CPP
 * @brief Solution for the "Soldier and Bananas" problem.
 * @author Your Name
 *
 * @details
 * This program calculates the amount of money a soldier needs to borrow to buy a certain number of bananas.
 * The cost of the bananas increases linearly: the first banana costs `k`, the second `2k`, and so on, up to the `w`-th banana which costs `w*k`.
 * The soldier initially has `n` dollars.
 *
 * @section algorithm
 * The total cost of `w` bananas is the sum of an arithmetic progression: `k * (1 + 2 + ... + w)`.
 * This sum is calculated using the formula for the sum of the first `w` integers: `w * (w + 1) / 2`.
 * The total cost is therefore `k * w * (w + 1) / 2`.
 * The amount to borrow is the difference between the total cost and the money the soldier has (`n`).
 * If the soldier has enough money, the amount to borrow is 0.
 *
 * @subsection complexity
 * - Time Complexity: O(1), as the solution involves a constant number of arithmetic operations.
 * - Space Complexity: O(1), as a fixed number of variables are used regardless of the input size.
 */
#include <iostream>
using namespace std;
int main(int argc, char const *argv[]) {
  // k: cost of the first banana
  // n: initial amount of money
  // w: number of bananas to buy
  int k, n, w;
  cin >> k >> n >> w;

  // Calculate the total cost of 'w' bananas.
  // The cost forms an arithmetic series: k + 2k + ... + wk = k * (1 + 2 + ... + w).
  // The sum of the first 'w' integers is w * (w + 1) / 2.
  long long total_cost = (long long)k * w * (w + 1) / 2;

  // Calculate the amount of money to borrow.
  // If the total cost is greater than the initial money 'n', the soldier needs to borrow the difference.
  // Otherwise, the soldier has enough money and doesn't need to borrow anything.
  long long borrow_amount = total_cost - n;

  // Output the amount to borrow. If the borrow_amount is negative (meaning the soldier has extra money), output 0.
  cout << (borrow_amount >= 0 ? borrow_amount : 0);
  return 0;
}