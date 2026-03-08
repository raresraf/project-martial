/**
 * @file 546A.CPP
 * @brief Solution for the "Soldier and Bananas" problem.
 *
 * @details
 * This program calculates the amount of money a soldier must borrow to buy a
 * specific number of bananas, where the price of each subsequent banana
 * increases in an arithmetic progression. The total cost is calculated using
 * a simple loop.
 *
 * @section
 * Algorithm: Iterative summation of an arithmetic series.
 * Time Complexity: O(W), where W is the number of bananas.
 * Space Complexity: O(1).
 *
 * @note An O(1) solution exists by using the mathematical formula for the sum
 * of an arithmetic series: `total_cost = k * w * (w + 1) / 2`.
 */
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
using namespace std;
int main() {
  // k: the cost of the first banana.
  // n: the initial amount of money the soldier possesses.
  // w: the number of bananas the soldier wants to buy.
  int k, n, w;
  cin >> k >> n >> w;

  // 'cost' will accumulate the total cost of all bananas.
  int cost = 0;
  /**
   * @brief This loop calculates the total cost by summing the price of each banana.
   *
   * Logic: For each banana from 1 to w, its cost is `i * k`. These individual
   * costs are added to the 'cost' accumulator. For example, if k=3 and w=4,
   * the calculation is 1*3 + 2*3 + 3*3 + 4*3.
   */
  for (int i = 1; i <= w; i++) {
    cost = cost + (i * k); // 1*3+2*3+3*3+4*3
  }

  /**
   * @brief Determine and print the amount of money to borrow.
   *
   * Logic: If the total cost exceeds the money on hand, the difference is
   * printed. Otherwise, no money needs to be borrowed, so 0 is printed.
   */
  if (cost > n) {
    cout << cost - n;
  } else {
    cout << 0;
  }
}
