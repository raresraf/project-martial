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
 * This is an arithmetic series, and the sum can be calculated directly as
 * `k * w * (w + 1) / 2`. After calculating the total cost, it subtracts the
 * initial money to find the deficit. If there is no deficit, the amount to
 * borrow is 0.
 *
 * @time_complexity O(W), where W is the number of bananas to buy, due to the loop.
 * A more optimal solution using the arithmetic series formula would be O(1).
 * @space_complexity O(1), as it only uses a few variables.
 */
#include <iostream>
using namespace std;
int main() {

  long long n; // The initial amount of money the person has.
  long long rst = 0; // Variable to accumulate the total cost. 'rst' likely means result.
  int k; // The cost of the first banana.
  int w; // The number of bananas to buy.
  
  // Read the input values.
  cin >> k >> n >> w;
  
  /**
   * @brief Loop to calculate the total cost of `w` bananas.
   * @invariant `rst` holds the sum of the costs of the first `i-1` bananas.
   */
  for (int i = 1; i <= w; i++) {
    // The cost of the i-th banana is i * k.
    rst += k * i;
  }
  
  /**
   * @brief Determine the amount to borrow.
   * @pre `rst` holds the total cost and `n` is the initial money.
   */
  if (rst - n > 0)
    // If the cost is greater than the money, print the difference.
    cout << rst - n;
  else
    // Otherwise, the person doesn't need to borrow money.
    cout << 0;
}
