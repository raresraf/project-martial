/**
 * @file 546A.CPP
 * @brief Solution to the "Soldier and Bananas" problem.
 *
 * This program calculates the amount of money a soldier needs to borrow to buy 'w' bananas.
 * The cost of the first banana is 'k', the second is '2k', and so on, up to 'wk'.
 * Algorithm: Summation of an arithmetic progression. The total cost is k * (1 + 2 + ... + w).
 *            The sum (1 + 2 + ... + w) can be calculated as w * (w + 1) / 2.
 *            The amount to borrow is the total cost minus the money the soldier already has ('n'),
 *            or 0 if the soldier has enough money.
 * Time Complexity: O(1) - constant time operations for input, calculation, and output.
 * Space Complexity: O(1) - constant space used for variables.
 */

#include <iostream>
using namespace std;

/**
 * @brief Main function to calculate the borrowed amount for bananas.
 *
 * Reads the cost of the first banana, initial money, and number of bananas.
 * Calculates the total cost and the amount to borrow, then prints the result.
 * @return 0 on successful execution.
 */
int main() {
  int k, n, w;
  // Block Logic: Read the input values for banana cost ('k'), initial money ('n'),
  //              and number of bananas ('w').
  // Pre-condition: User provides three positive integers.
  // Invariant: k, n, and w will hold the integer values.
  cin >> k >> n >> w;

  // Block Logic: Calculate the total cost, the amount to borrow, and print the result.
  // Total cost = k * (1 + 2 + ... + w) = k * w * (w + 1) / 2.
  // Amount to borrow = max(total_cost - n, 0).
  // Post-condition: The non-negative amount of money to borrow is printed to standard output.
  cout << max(w * (w + 1) * k / 2 - n, 0);
}
