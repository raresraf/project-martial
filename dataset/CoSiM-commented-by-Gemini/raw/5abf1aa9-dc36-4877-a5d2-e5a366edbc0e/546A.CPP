/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem from Codeforces (546A).
 * @details The problem is to calculate how much money a soldier needs to borrow to buy
 * a specific number of bananas, where the price increases for each banana purchased.
 *
 * @algorithm
 * The cost of the bananas forms an arithmetic progression. The total cost is the sum
 * of `k*1, k*2, ..., k*w`.
 * 1. The solution reads the initial banana cost (`k`), the soldier's money (`n`),
 *    and the number of bananas he wants (`w`).
 * 2. It iterates from 1 to `w`, summing up the cost of each banana (`k * i`) into a
 *    `total` variable. This calculates the sum of the arithmetic series.
 * 3. After the loop, it compares the `total` cost with the money `n` the soldier has.
 * 4. If the total cost is less than his money, he doesn't need to borrow, so 0 is printed.
 * 5. If the total cost is greater than or equal to his money, the difference (`total - n`)
 *    is calculated and printed. If total equals n, this correctly results in 0.
 *
 * @complexity
 * Time Complexity: O(W), where W is the number of bananas, due to the single loop
 * for calculating the total cost.
 * Space Complexity: O(1), using a fixed number of variables.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  /**
   * @variable k: The cost of the first banana.
   * @variable n: The initial amount of money the soldier possesses.
   * @variable w: The number of bananas the soldier wants to buy.
   * @variable total: The calculated total cost for all `w` bananas.
   * @variable sol: The solution, i.e., the amount of money to be borrowed.
   */
  int k, n, w;
  int total = 0;
  int sol;

  cin >> k >> n >> w;
  
  /**
   * @block
   * @brief This loop calculates the total cost of `w` bananas.
   * The cost of the i-th banana is `k * i`, forming an arithmetic progression.
   */
  for (int i = 1; i <= w; i++) {
    total += k * i;
  }
  
  /**
   * @block
   * @brief Determines the amount of money the soldier needs to borrow.
   */
  // Pre-condition: Check if the soldier has more money than the total cost.
  if (total <= n) {
    cout << "0" << endl;
  } else {
    // If cost is greater than his money, calculate the deficit.
    sol = total - n;
    cout << sol << endl;
  }
  return 0;
}