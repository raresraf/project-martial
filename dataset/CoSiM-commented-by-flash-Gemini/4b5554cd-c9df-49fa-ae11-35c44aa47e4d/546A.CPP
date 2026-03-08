/**
 * @file 546A.CPP
 * @brief Calculates the amount of money needed to borrow to buy a specified number of bananas.
 *        The cost of each subsequent banana increases linearly.
 *
 * Algorithm: Iterative summation of banana costs.
 * Time Complexity: O(bananas) as it iterates through each banana to sum its cost.
 */
#include <iostream>

using namespace std;

/**
 * @brief Main function where the banana cost calculation logic resides.
 * @pre None.
 * @post Prints the amount of money to borrow, or 0 if no borrowing is needed.
 */
int main() {
  int cost, money, bananas, borrow = 0;
  // Functional Utility: Reads the initial cost of one banana, the initial money, and the number of bananas.
  cin >> cost >> money >> bananas;

  /**
   * Block Logic: Calculates the total cost of purchasing 'bananas'.
   * Invariant: In each iteration, 'borrow' accumulates the cost of the current banana (cost * i).
   * The cost of the i-th banana is 'cost * i'.
   */
  for (int i = 1; i <= bananas; i++) {
    borrow = borrow + (cost * i);
  }

  /**
   * Block Logic: Determines the amount of money to be borrowed.
   * Pre-condition: 'borrow' holds the total cost of all bananas, and 'money' holds the initial cash.
   * Invariant: If 'borrow' is greater than 'money', the difference represents the amount to borrow.
   *            Otherwise, no borrowing is needed.
   */
  if (borrow > money) {
    borrow = borrow - money;
    // Output the calculated amount to borrow.
    cout << borrow;
  } else {
    // Output 0, indicating no money needs to be borrowed.
    cout << '0';
  }
  // Indicates successful program execution.
  return 0;
}