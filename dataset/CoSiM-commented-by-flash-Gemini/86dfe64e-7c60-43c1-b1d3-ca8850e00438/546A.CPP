/**
 * @file 546A.CPP
 * @brief Solution to a competitive programming problem: "Soldier and Bananas".
 *
 * Calculates the amount of money a soldier needs to borrow to buy a specified number of bananas,
 * where the cost of each successive banana increases linearly.
 *
 * Algorithm: Linear summation of banana costs and conditional output based on available money.
 * Time Complexity: O(W), where W is the number of bananas.
 */
#include <iostream>
using namespace std;

/**
 * @brief Main function to calculate the money needed for bananas.
 * Reads initial money, cost of first banana, and number of bananas to buy.
 * Calculates total cost and outputs the amount to be borrowed or 0 if sufficient.
 * @return 0 on successful execution.
 */
int main() {

  /**
   * Declares variables:
   * 'k': The cost of the first banana and the increment for subsequent bananas.
   * 'n': The initial amount of money the soldier possesses.
   * 'w': The total number of bananas the soldier wishes to buy.
   * 'amt': Accumulates the total cost of purchasing 'w' bananas.
   * Reads the values for 'k', 'n', and 'w' from standard input.
   * Pre-condition: Standard input provides three integers representing k, n, and w.
   */
  int k, n, w;
  int amt = 0;
  cin >> k >> n >> w;

  /**
   * Block Logic: Calculates the total cost of 'w' bananas.
   * The cost of the i-th banana is i * k.
   * Invariant: 'amt' accurately accumulates the sum of costs for all purchased bananas.
   * Pre-condition: 'k' and 'w' have been initialized.
   */
  for (int i = 1; i <= w; i++)
    amt += i * k;

  /**
   * Block Logic: Determines the amount of money the soldier needs to borrow.
   * If the total cost ('amt') exceeds the available money ('n'), the difference is the amount to borrow.
   * Otherwise, the soldier has enough money or more, and 0 is outputted.
   * Post-condition: The required borrowing amount (or 0) is printed to standard output.
   */
  if (amt - n > 0)
    cout << amt - n << endl;
  else
    cout << 0 << endl;
  return 0;
}
