/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem.
 *
 * This program calculates the amount of money a soldier needs to borrow to buy
 * a certain number of bananas. The cost of the bananas increases linearly: the
 * first costs k, the second 2k, and so on.
 *
 * Algorithm:
 * 1. Read the initial banana cost (k), the soldier's money (n), and the number
 *    of bananas to buy (w).
 * 2. Calculate the total cost of 'w' bananas. This is the sum of an arithmetic
 *    series: k * (1 + 2 + ... + w). The code computes this sum using a loop.
 * 3. If the soldier's money 'n' is sufficient, the amount to borrow is 0.
 * 4. Otherwise, the amount to borrow is the total cost minus 'n'.
 *
 * Time Complexity: O(W), where W is the number of bananas, due to the loop used
 * for calculating the total cost.
 * Space Complexity: O(1).
 *
 * @note The total cost can also be calculated in O(1) using the formula for the
 * sum of an arithmetic series: cost = k * w * (w + 1) / 2.
 */
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

int main() {
  int k; // Cost of the first banana.
  int n; // Initial money the soldier has.
  int w; // Number of bananas to buy.
  int total_cost = 0;
  int next_banana; // Declared but not used.
  cin >> k >> n >> w;

  /**
   * Block Logic: Calculate the total cost of 'w' bananas.
   * The cost forms an arithmetic progression. This loop sums the cost of
   * each banana from the 1st to the w-th.
   */
  for (int i = 1; i <= w; ++i) {
    total_cost += k * i;
  }
  
  /**
   * Block Logic: Determine the amount to borrow.
   * Compare the total cost with the money available.
   */
  if (n >= total_cost) {
    // If the soldier has enough money, no borrowing is needed.
    cout << 0;
  } else {
    // Otherwise, calculate and print the deficit.
    cout << total_cost - n;
  }
}
