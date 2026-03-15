/**
 * @file 546A.CPP
 * @brief Solution to the "Soldier and Bananas" problem.
 * @details This program calculates the amount of money a soldier needs to borrow to buy `w` bananas,
 * where the cost increases linearly for each banana.
 *
 * @algorithm
 * 1. Reads the initial cost of one banana `k`, the soldier's initial money `n`, and the desired number of bananas `w`.
 * 2. It calculates the total cost by iterating from 1 to `w` and summing up the cost of each banana (`i * k`). This is a direct summation of an arithmetic progression.
 * 3. It then determines the amount to borrow. If the calculated `sum` of costs exceeds the soldier's money `n`, the difference (`sum - n`) is printed.
 * 4. If the soldier has enough money, 0 is printed.
 *
 * @complexity
 * - Time: O(W), where W is the number of bananas to buy, due to the linear summation loop.
 * - Space: O(1), as the program uses a fixed amount of memory.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // k: The cost of the first banana.
  // n: The initial amount of money the soldier possesses.
  // w: The total number of bananas the soldier intends to buy.
  // sum: The calculated total cost for all `w` bananas.
  int k, n, w, sum = 0;
  cin >> k >> n >> w;

  /**
   * @brief This loop calculates the total cost for `w` bananas.
   * The cost of the i-th banana is `i * k`. The loop sums these costs.
   * Invariant: After iteration `i`, `sum` holds the total cost of the first `i` bananas.
   */
  for (int i = 1; i <= w; i++) {
    sum += (k * i);
  }

  /**
   * @brief This block determines and prints the amount of money to borrow.
   * If the total cost `sum` is greater than the soldier's money `n`, the deficit is printed.
   * Otherwise, the soldier has enough money, and 0 is printed.
   */
  if ((sum - n) > 0)
    cout << (sum - n);
  else
    cout << "0";
  return 0;
}