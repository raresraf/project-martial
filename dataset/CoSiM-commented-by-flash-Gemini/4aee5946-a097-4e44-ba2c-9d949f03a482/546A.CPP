/**
 * @4aee5946-a097-4e44-ba2c-9d949f03a482/546A.CPP
 * @brief This program calculates the total cost of purchasing a series of items
 *        with increasing prices and determines the amount of money needed or remaining.
 * Algorithm: Arithmetic series sum for total cost, followed by a simple subtraction and absolute value.
 * Time Complexity: O(1) as it involves a fixed number of arithmetic operations.
 * Space Complexity: O(1) for storing a few integer variables.
 */
#include <iostream>
using namespace std;
int main() {
  // Declare variables:
  // k: The cost of the first banana.
  // w: The number of bananas the soldier wants to buy.
  int k, w;
  // n: The initial amount of money the soldier has (using long to handle potentially larger values).
  // amountToBuyWBananas: Stores the total calculated cost for 'w' bananas.
  long n, amountToBuyWBananas = 0;
  // Functional Utility: Read the input values for k, n, and w.
  cin >> k >> n >> w;
  // Block Logic: Calculate the total cost of 'w' bananas.
  // The cost of the i-th banana is i * k.
  // The total cost is k * (1 + 2 + ... + w) = k * w * (w + 1) / 2 (sum of an arithmetic series).
  amountToBuyWBananas = ((w * (w + 1)) / 2) * k;
  // Pre-condition: Check if the soldier needs to borrow money (i.e., initial money is less than total cost).
  if (n - amountToBuyWBananas < 0)
    // Action: If money is needed, output the absolute difference (amount to borrow).
    cout << abs(n - amountToBuyWBananas);
  else {
    // Action: Otherwise, output 0 (no money needed to borrow).
    cout << "0";
  }
}