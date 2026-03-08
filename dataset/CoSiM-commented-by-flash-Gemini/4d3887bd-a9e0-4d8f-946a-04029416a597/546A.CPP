/**
 * @file 546A.CPP
 * @brief Calculates the amount of money needed to borrow to buy a specified number of items (e.g., bananas).
 *        The cost of each subsequent item increases linearly: 1st item costs 'k', 2nd '2*k', ..., w-th 'w*k'.
 *
 * Problem Name: Giga Banana (Codeforces 546A)
 * Algorithm: Iterative summation of item costs.
 * Time Complexity: O(W) where W is the number of items, due to a linear pass through the items.
 * Space Complexity: O(1) as it uses a constant number of variables.
 */
#include <iostream> // Required for input/output operations (cin, cout).
using namespace std;

// Global variables for cost of first item, initial money, number of items, and accumulated total cost.
int k, n, w, pas = 0;

/**
 * @brief Main function to calculate the money to be borrowed for purchasing items.
 * @pre Input consists of three integers: 'k' (cost of first item), 'n' (initial money), and 'w' (number of items).
 * @post Prints the amount of money to borrow, or 0 if no borrowing is needed.
 */
int main() {
  // Functional Utility: Reads the input values for cost of first item, initial money, and number of items.
  cin >> k >> n >> w;

  /**
   * Block Logic: Calculates the total cost of purchasing 'w' items using an iterative sum.
   * Invariant: In each iteration, 'pas' accumulates the cost of the current item (k * i).
   *            The cost of the i-th item is 'k * i'.
   */
  for (int i = 1; i <= w; i++) {
    pas += k * i; // Adds the cost of the i-th item to the total.
  }

  /**
   * Block Logic: Determines the amount of money to be borrowed.
   * Pre-condition: 'pas' holds the total calculated cost, and 'n' holds the initial money.
   * Invariant: If 'pas' (total cost) is less than 'n' (initial money), no borrowing is needed (result is 0).
   *            Otherwise, the difference ('pas - n') is the amount to borrow.
   */
  if (pas - n < 0) {
    cout << 0; // If total cost is less than initial money, print 0 (no borrowing).
  } else {
    cout << pas - n; // Otherwise, print the amount to borrow.
  }
  // No explicit return 0; is needed for main in C++ if it reaches the end.
  // However, for competitive programming, it's often good practice to include it.
}