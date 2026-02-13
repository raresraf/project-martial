/**
 * @file 546A.CPP
 * @brief Solution for the "Soldier and Bananas" problem (Codeforces 546A).
 * @author @Nishant
 *
 * @details This program calculates the amount of money a soldier needs to borrow
 * to buy a specific number of bananas. The cost of the bananas increases
 * linearly: the first costs `k`, the second `2k`, and so on.
 *
 * Algorithm (Simulation Approach):
 * 1. Read the initial banana cost (`k`), the soldier's initial money (`n`),
 *    and the number of bananas to buy (`w`).
 * 2. The first loop simulates the soldier buying bananas one by one as long as
 *    he can afford them, deducting the cost from his money `n`.
 * 3. When he can no longer afford a banana, the second loop calculates the total
 *    cost of the remaining bananas. This is the `needed` amount.
 * 4. The final amount to borrow is the `needed` amount minus the soldier's
 *    remaining money. If this is negative or zero, he needs to borrow nothing.
 *
 * @note A more direct mathematical approach exists:
 * Total cost = k * (1 + 2 + ... + w) = k * w * (w + 1) / 2.
 * Amount to borrow = max(0, Total cost - n).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // k: cost of the first banana
  // n: initial money the soldier has
  // w: total number of bananas to buy
  int k, n, w;
  cin >> k >> n >> w;
  
  // `i` tracks the current banana number (and thus the cost multiplier).
  int i = 1;
  
  /**
   * Block Logic: This loop simulates the soldier buying bananas for as long
   * as he has enough money.
   * @warning Logical Flaw: The condition `n - i * k > 0` should be `>= 0`.
   * If the soldier has exactly enough money, he can still buy the banana.
   */
  while (n > 0 && w >= 0) {
    if (n - i * k > 0) {
      n -= i * k; // Deduct cost from money.
      i++;        // Move to the next banana.
      w--;        // Decrement bananas left to buy.
    } else {
      break; // Not enough money, stop buying.
    }
  }
  
  // `needed` will accumulate the cost of the remaining bananas.
  int needed = 0;
  
  /**
   * Block Logic: This loop calculates the total cost of all the bananas the
   * soldier could not afford with his initial money.
   */
  while (w > 0) {
    needed += i * k;
    w--;
    i++;
  }
  
  /**
   * Decision Logic: Calculate the final amount to borrow.
   * This is the cost of the remaining bananas (`needed`) minus the soldier's
   * leftover money (`n`).
   */
  if (needed > 0) {
    cout << needed - n;
  } else {
    // If he could afford all bananas, he needs to borrow 0.
    cout << 0;
  }
  return 0;
}
