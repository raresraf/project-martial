/**
 * @file 546A.CPP
 * @brief Calculates the amount of money to borrow to buy a number of bananas.
 *
 * This program solves a problem where a person buys a specific number of bananas (`w`).
 * The cost of the first banana is `k`, the second is `2k`, and so on, up to the w-th
 * banana which costs `w*k`. The program calculates the total cost and determines
 * how much money the person needs to borrow, given their initial amount of money (`n`).
 *
 * @algorithm The program iteratively calculates the total cost by summing up the
 * cost of each banana from 1 to `w`. The cost of the i-th banana is `i * k`.
 * This approach is functionally an arithmetic series summation.
 * A more direct calculation would be `k * w * (w + 1) / 2`.
 *
 * @time_complexity O(W), where W is the number of bananas, due to the loop.
 * A more optimal solution using the arithmetic series formula would be O(1).
 * @space_complexity O(1), as it only uses a few global variables.
 */
#include <iostream>
using namespace std;

// Global variables to hold the problem parameters and result.
int k;       // The cost of the first banana.
int n;       // The initial amount of money the person has.
int w;       // The number of bananas to buy.
int pas = 0; // Accumulator for the total cost.

int main() {
  // Read the input values for k, n, and w.
  cin >> k >> n >> w;
  
  /**
   * @brief Loop to calculate the total cost of `w` bananas.
   * @invariant `pas` holds the sum of the costs of the first `i-1` bananas.
   */
  for (int i = 1; i <= w; i++) {
    // Add the cost of the i-th banana to the total.
    pas += k * i;
  }
  
  /**
   * @brief Determine the amount of money to borrow.
   * @pre `pas` holds the total cost and `n` is the initial money.
   */
  if (pas - n < 0) {
    // If the person has enough or more money than needed, no borrowing is required.
    cout << 0;
  } else {
    // If the cost exceeds the money, print the difference.
    cout << pas - n;
  }
}