/**
 * @file 546A.CPP
 * @brief Logic for calculating purchase deficits based on arithmetic price progression.
 * 
 * Logic: Calculates the total cost of buying 'w' items where the i-th item costs 
 * i * k. The total price is the sum of an arithmetic series: k * (1 + 2 + ... + w). 
 * The output is the required borrowing amount (Deficit), clamped at a minimum of 0.
 * 
 * Algorithm: Closed-form arithmetic series summation.
 * Formula: Total Cost = k * [w * (w + 1) / 2].
 * 
 * Time Complexity: $O(1)$ constant time calculation.
 * Space Complexity: $O(1)$ constant auxiliary space.
 */

#include <iostream>
using namespace std;

int main() {
  long long k, n, w, total_price, borrow_amount;
  
  /**
   * Functional Utility: Reads purchase parameters.
   * k: Price multiplier for the first item.
   * n: Available capital (initial budget).
   * w: Target quantity of items to purchase.
   */
  if (!(cin >> k >> n >> w)) return 0;

  /**
   * Logic: Sum of the first 'w' integers multiplied by k.
   * Uses the standard formula for triangular numbers.
   */
  total_price = k * ((w * (w + 1)) / 2);
  
  // Functional Utility: Calculates the net deficit.
  borrow_amount = total_price - n;
  
  /**
   * Logic: Non-negative clamping.
   * If current capital exceeds total price, no borrowing is required.
   */
  if (borrow_amount <= 0) {
    borrow_amount = 0;
  }
  
  // Output the required financial supplement.
  cout << borrow_amount << endl;
  
  return 0;
}
