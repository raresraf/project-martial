/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem using an iterative approach.
 * @details This program calculates the amount of money a soldier needs to borrow
 * to buy `w` bananas, where the cost increases linearly for each banana.
 *
 * @algorithm
 * 1. Read the initial cost `k`, the soldier's money `n`, and the number of
 *    bananas `w`.
 * 2. Use a loop to iterate from 1 to `w`, calculating the cost of each banana
 *    (`k * i`) and adding it to a running total.
 * 3. After the loop, the total cost is known.
 * 4. Subtract the soldier's money `n` from the total cost. If the result is
 *    positive, this is the amount to borrow. If it's zero or negative, the
 *    soldier has enough money, and nothing needs to be borrowed.
 *
 * @complexity
 * Time Complexity: O(W), where W is the number of bananas, due to the loop.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;

// Global variables to store the problem inputs and the calculated total cost.
// While this works for a small program, using local variables within main()
// is generally better practice.
int k, n, w;
long long total_cost = 0; // Use long long to prevent overflow.

int main() {
  cin >> k >> n >> w;

  /**
   * @block
   * @brief Iteratively calculate the total cost of all bananas.
   */
  for (int i = 1; i <= w; i++) {
    total_cost += k * i;
  }

  /**
   * @block
   * @brief Determine and print the amount of money to borrow.
   */
  if (total_cost - n < 0) {
    // If the cost is less than or equal to the money available, borrow 0.
    cout << 0;
  } else {
    // Otherwise, the amount to borrow is the difference.
    cout << total_cost - n;
  }
}
