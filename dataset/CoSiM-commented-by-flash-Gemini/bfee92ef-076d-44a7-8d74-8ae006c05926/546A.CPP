/**
 * @file 546A.CPP
 * @brief Calculates the amount of money needed to buy 'w' bananas,
 *        where the cost of the i-th banana is 'i * k', using an optimized formula.
 * Algorithm: Direct calculation using the sum of an arithmetic progression formula.
 *            The cost of 'w' bananas is k * (1 + 2 + ... + w) = k * w * (w + 1) / 2.
 * Time Complexity: O(1) as it involves a fixed number of arithmetic operations.
 * Space Complexity: O(1) for storing a few integer variables.
 */
#include <iostream>

using namespace std;

/**
 * @brief Main function to calculate the money required based on banana costs.
 * Functional Utility: Determines the deficit (money to borrow) when purchasing
 *                     a specific number of bananas with increasing costs.
 */
int main() {
  // Variable declarations:
  // k: Cost of the first banana (and multiplier for subsequent bananas).
  // n: Initial amount of money the soldier has.
  // w: Number of bananas the soldier wants to buy.
  // a: Stores the calculated money deficit (or surplus if negative).
  int k, n, w, a;
  
  // Block Logic: Reads the problem parameters (initial cost, money, number of bananas).
  cin >> k >> n >> w;
  
  // Block Logic: Calculates the total cost for 'w' bananas using the sum of an arithmetic
  //              progression formula: sum(1 to w) = w * (w + 1) / 2.
  //              Then, subtracts the available money 'n' to find the deficit 'a'.
  // Functional Utility: Efficiently computes the total money required.
  a = ((((w * (w + 1)) / 2) * k) - n);
  
  // Block Logic: Outputs the money needed. If 'a' is positive, it's the deficit.
  //              If 'a' is zero or negative (meaning enough or surplus money), outputs 0.
  // Functional Utility: Provides the final answer as per problem requirements.
  if (a > 0) {
    cout << a << endl;
  } else {
    cout << 0 << endl;
  }
  
  return 0; // Indicates successful program termination.
}
