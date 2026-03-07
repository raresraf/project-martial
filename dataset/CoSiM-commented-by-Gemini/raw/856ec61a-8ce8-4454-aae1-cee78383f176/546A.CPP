/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem.
 *
 * @details
 * A soldier buys 'w' bananas. The first banana costs 'k', the second '2k', and so on,
 * up to the w-th banana costing 'w*k'. The soldier has 'n' dollars. This program
 * calculates the amount of money the soldier must borrow to buy all the bananas.
 * If the soldier has enough money, the amount to borrow is 0.
 *
 * Algorithm:
 * 1. Read the initial cost 'k', the soldier's initial money 'n', and the number
 *    of bananas to buy 'w'.
 * 2. Calculate the total cost of 'w' bananas. The cost is the sum of an arithmetic
 *    progression: k * (1 + 2 + ... + w).
 * 3. The sum of the first 'w' integers is w*(w+1)/2. So, total cost = k * w * (w+1) / 2.
 *    The program calculates this iteratively.
 * 4. Calculate the amount to borrow by subtracting the soldier's money 'n' from the total cost.
 * 5. If the result is positive, that is the amount to borrow. Otherwise, the soldier
 *    has enough money, and the amount to borrow is 0.
 *
 * Time Complexity: O(W), where W is the number of bananas, due to the loop to calculate the total cost.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;

int main() {
  int k, n, w;
  // k: cost of the first banana
  // n: initial amount of money the soldier has
  // w: number of bananas to buy
  cin >> k >> n >> w;

  int sum = 0; // Variable to store the total cost of the bananas.
  
  /**
   * @brief Loop to calculate the total cost of 'w' bananas.
   * The cost of the i-th banana is i*k.
   * Invariant: `sum` holds the cost of the first `i-1` bananas.
   */
  for (int i = 1; i <= w; i++) {
    sum = sum + i * k;
  }
  
  // Calculate the amount of money the soldier needs to borrow.
  int res = sum - n;

  /**
   * @brief Print the result.
   * If `res` is positive, it's the amount to borrow.
   * If `res` is zero or negative, the soldier has enough money, so print 0.
   */
  if (res > 0)
    cout << res;
  else
    cout << 0;
  return 0;
}