/**
 * @file 546A.CPP
 * @brief Solution for Codeforces problem 546A "Soldier and Bananas".
 * @details This problem requires calculating the amount of money a soldier needs to borrow to buy a certain number of bananas with a linearly increasing cost.
 *
 * @section Algorithm
 * The cost of the bananas forms an arithmetic progression. The cost of the first banana is `k`, the second is `2k`, and so on, up to the `w`-th banana which costs `w*k`.
 * The total cost is the sum of this arithmetic series: `k * (1 + 2 + ... + w)`.
 * This sum can be calculated using the formula for the sum of the first `w` positive integers: `k * w * (w + 1) / 2`.
 * The code, however, uses a loop to compute this sum iteratively.
 *
 * 1. Read the input values: `k` (initial cost), `n` (initial money), `w` (number of bananas).
 * 2. Calculate the total cost of `w` bananas by iterating from 1 to `w` and summing up `k * i` at each step.
 * 3. If the total cost is greater than the money the soldier has (`n`), the amount to borrow is the difference: `total_cost - n`.
 * 4. Otherwise, the soldier has enough money, so the amount to borrow is 0.
 *
 * @subsection Complexity
 * - Time Complexity: O(w) because of the loop that iterates `w` times.
 * - Space Complexity: O(1) as it uses a constant amount of extra space.
 */
#include <bits/stdc++.h>

using namespace std;
int main(int argc, char const *argv[]) {
  // k: The cost of the first banana.
  // n: The initial amount of money the soldier has.
  // w: The number of bananas the soldier wants to buy.
  int k, n, w;
  cin >> k;
  cin >> n;
  cin >> w;

  // money: This variable will accumulate the total cost of the bananas.
  int money = 0;
  // amt: This variable holds the cost of the current banana in each iteration.
  int amt = 1;

  /**
   * @brief This loop calculates the total cost to purchase `w` bananas.
   * Pre-condition: `money` is 0. `k` and `w` are the initial cost and number of items.
   * Invariant: At the end of each iteration `i`, the `money` variable holds the sum of the costs for the first `i` bananas.
   * The cost of the `i`-th banana is `k * i`.
   */
  for (int i = 1; i <= w; i++) {
    amt = k * i;
    money = money + amt;
  }

  /**
   * @brief Determines the amount of money to borrow.
   * If the total `money` required is greater than the initial amount `n` the soldier has,
   * the difference is calculated, which is the amount to be borrowed.
   * Otherwise, the soldier has enough money, and no borrowing is needed.
   */
  if (money > n) {
    int x = money - n;
    cout << x;
  }

  else {
    cout << "0";
  }

  return 0;
}