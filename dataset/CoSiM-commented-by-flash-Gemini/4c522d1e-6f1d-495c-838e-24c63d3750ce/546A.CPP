/**
 * @file 546A.CPP
 * @brief Calculates the amount of money needed to borrow to buy a specified number of items (bananas).
 *        The cost of each subsequent item increases linearly.
 *
 * Algorithm: Iterative summation of item costs.
 * Time Complexity: O(W) where W is the number of items, as it iterates through each item to sum its cost.
 * Space Complexity: O(1) as only a few variables are used to store state.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.

using namespace std;

/**
 * @brief Main function where the item cost calculation logic resides.
 * @pre Input consists of three integers: 'k' (cost of the first item), 'n' (initial money), and 'w' (number of items).
 * @post Prints the amount of money to borrow, or 0 if no borrowing is needed.
 */
int main() {
  int k, n, w; // 'k': cost of the first item, 'n': initial money, 'w': number of items to buy.
  // Functional Utility: Reads the input values for cost, initial money, and number of items.
  cin >> k >> n >> w;

  int sum = 0; // Accumulates the total cost of purchasing 'w' items.

  /**
   * Block Logic: Calculates the total cost of purchasing 'w' items.
   * Invariant: In each iteration, 'sum' accumulates the cost of the current item (i * k).
   * The cost of the i-th item is 'k * i'.
   */
  for (int i = 1; i <= w; i++)
    sum += i * k; // Add the cost of the i-th item.

  /**
   * Block Logic: Determines the amount of money to be borrowed.
   * Pre-condition: 'sum' holds the total calculated cost, and 'n' holds the initial money.
   * Invariant: If 'sum' is greater than or equal to 'n', the difference represents the amount to borrow.
   *            Otherwise, no borrowing is needed.
   */
  if (sum >= n)
    cout << sum - n << endl; // Output the calculated amount to borrow.
  else
    cout << "0" << endl; // Output 0, indicating no money needs to be borrowed.

  // Indicates successful program execution.
  return 0;
}
