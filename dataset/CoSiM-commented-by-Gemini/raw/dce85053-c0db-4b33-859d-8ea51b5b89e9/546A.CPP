#include <iostream>

/**
 * @file 546A.CPP
 * @brief Solution for a computational problem involving calculating the cost of items with progressively increasing prices.
 *
 * This program calculates the total cost of buying a certain number of bananas,
 * where the price of each subsequent banana increases linearly. It then
 * determines the amount of money that needs to be borrowed if the initial
 * amount is insufficient.
 *
 * Problem Statement:
 * A soldier wants to buy `w` bananas. The first banana costs `k` dollars, the
 * second costs `2k`, the third `3k`, and so on. The soldier has `n` dollars.
 * Calculate the amount of money the soldier must borrow to buy all `w` bananas.
 * If the soldier has enough money, the amount to borrow is 0.
 *
 * Algorithm:
 * The total cost is the sum of an arithmetic progression: k + 2k + ... + wk.
 * This can be expressed as k * (1 + 2 + ... + w).
 * The sum of the first `w` positive integers is `w * (w + 1) / 2`.
 * Therefore, the total cost is `k * w * (w + 1) / 2`.
 *
 * The current implementation uses an iterative approach to calculate the cost.
 * Time Complexity: O(w) for the `getCost` function.
 * Space Complexity: O(1).
 */

using namespace std;

/**
 * @brief Calculates the total cost of buying `w` items with linearly increasing cost.
 *
 * The cost of the i-th item (1-indexed) is `i * k`. This function sums the
 * costs for all `w` items.
 *
 * @param w The number of items to buy.
 * @param k The base cost of the first item.
 * @return The total cost.
 * @note A more efficient O(1) approach would be to use the arithmetic series
 *       formula: cost = k * (w * (w + 1) / 2).
 */
int getCost(int w, int k) {
  int cost = 0;
  // Invariant: 'cost' holds the sum of the costs of the first 'i' bananas.
  for (int i = 0; i < w; i++) {
    // The cost of the (i+1)-th banana is (i+1) * k.
    cost += (i + 1) * k;
  }
  return cost;
}

int main() {
  int k, n, w;
  // k: cost of the first banana
  // n: initial amount of money
  // w: number of bananas to buy
  cin >> k >> n >> w;

  // Calculate the total cost of w bananas.
  int cost = getCost(w, k);

  // Determine if the soldier needs to borrow money.
  if (n >= cost) {
    // If the initial money is enough, no borrowing is needed.
    cout << 0 << endl;
  } else {
    // Otherwise, calculate the deficit.
    cout << (cost - n) << endl;
  }

  return 0;
}
