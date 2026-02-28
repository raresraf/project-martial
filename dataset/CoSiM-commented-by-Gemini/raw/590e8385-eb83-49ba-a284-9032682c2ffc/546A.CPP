/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem from Codeforces (546A).
 * @details The problem requires calculating the amount of money a soldier needs to borrow
 * to buy a certain number of bananas, where the cost of each banana increases sequentially.
 *
 * @algorithm
 * The cost of the bananas forms an arithmetic progression. The cost of the first banana is `k`,
 * the second is `2k`, and so on, up to the `w`-th banana which costs `w*k`.
 * The total cost is the sum of this arithmetic series: `k * (1 + 2 + ... + w)`.
 * 1. The solution reads the cost of the first banana (`k`), the soldier's initial money (`n`),
 *    and the number of bananas to buy (`w`).
 * 2. It iterates from 1 to `w`, calculating the cost of each banana (`k * i`) and adding it
 *    to a running total `sum`. (Alternatively, one could use the formula for the sum of an
 *    arithmetic series: `total_cost = k * w * (w + 1) / 2`).
 * 3. After computing the total cost, it checks if the soldier's initial money `n` is sufficient.
 * 4. If the total cost `sum` is greater than `n`, it prints the difference (`sum - n`),
 *    which is the amount to be borrowed.
 * 5. Otherwise, if the soldier has enough money, it prints 0.
 *
 * @complexity
 * Time Complexity: O(W), where W is the number of bananas to buy, due to the loop that
 * calculates the total cost.
 * Space Complexity: O(1), as it uses a fixed number of variables.
 */
#include <bits/stdc++.h>

using namespace std;
#define ll long long
int main() {

  /**
   * @variable k: The cost of the first banana.
   * @variable n: The initial amount of money the soldier has (long long to handle large values).
   * @variable w: The number of bananas the soldier wants to buy.
   */
  int k, w;
  ll n;
  /**
   * @variable sum: The total cost of all `w` bananas.
   */
  ll sum = 0;

  // Read the input values.
  cin >> k >> n >> w;
  
  /**
   * @block
   * @brief This loop calculates the total cost for `w` bananas.
   * The cost of the i-th banana is `k * i`.
   * @invariant `sum` holds the total cost of the first `i-1` bananas.
   */
  for (int i = 1; i <= w; i++) {
    sum += k * i;
  }
  
  /**
   * @block
   * @brief Determines the amount of money to borrow.
   */
  // Pre-condition: Check if the total cost exceeds the money available.
  if (sum - n > 0)
    // If so, print the deficit.
    cout << sum - n;
  else
    // Otherwise, no money needs to be borrowed.
    cout << "0";
    
  return 0;
}
