/**
 * @file 546A.CPP
 * @brief Solution to Codeforces problem 546A (Soldier and Bananas).
 * @author Solved by: [User who submitted the code]
 *
 * @details
 * This program calculates the amount of money a soldier needs to borrow.
 * The soldier wants to buy `w` bananas. The first banana costs `k`, the second
 * `2k`, and so on. The soldier has `n` dollars.
 *
 * @section algorithm
 * This implementation calculates the total cost of the bananas using a loop
 * and then determines the deficit.
 *
 * The algorithm is as follows:
 * 1. Read the initial cost `k` (here, `cost`), initial money `n` (`money`),
 *    and number of bananas `w` (`bananas`).
 * 2. Initialize a `total_cost` variable to 0.
 * 3. Loop from `i = 1` to `w`. In each iteration, add the cost of the i-th
 *    banana (`i * k`) to the `total_cost`.
 * 4. After the loop, compare the `total_cost` with the `money` the soldier has.
 * 5. If `total_cost` is greater than `money`, the amount to borrow is
 *    `total_cost - money`.
 * 6. Otherwise, the amount to borrow is 0.
 *
 * This iterative approach is correct but has a linear time complexity,
 * whereas a more optimal solution using the arithmetic series formula
 * would be constant time.
 *
 * @section complexity
 * - Time Complexity: O(W), where W is the number of bananas to buy, due to the
 *   loop used to calculate the total cost.
 * - Space Complexity: O(1).
 */

#include <iostream>

using namespace std;

int main() {
  // cost: The cost of the first banana (k).
  // money: The soldier's initial money (n).
  // bananas: The number of bananas to buy (w).
  long long cost, money, bananas;
  cin >> cost >> money >> bananas;

  // Use 'total_cost' to accumulate the price of all bananas.
  long long total_cost = 0;

  /**
   * @brief Loop to calculate the total cost of w bananas.
   *
   * This loop iterates from the first banana to the w-th banana, summing
   * up their increasing costs.
   */
  for (int i = 1; i <= bananas; i++) {
    total_cost = total_cost + (cost * i);
  }

  // Check if the total cost exceeds the money the soldier has.
  if (total_cost > money) {
    // If so, the amount to borrow is the difference.
    long long borrow_amount = total_cost - money;
    cout << borrow_amount;
  } else {
    // Otherwise, the soldier has enough money.
    cout << 0;
  }

  return 0;
}
