/**
 * @file 546A.CPP
 * @brief Solution for Codeforces Problem 546A - "Soldier and Bananas".
 * This program calculates the amount of money a soldier needs to borrow to buy a certain
 * number of bananas, given the cost progression and the money the soldier already has.
 * The cost of the i-th banana is i*k.
 * Algorithm: Direct arithmetic calculation of the sum of an arithmetic progression.
 * Time Complexity: O(1) - Involves a fixed number of arithmetic operations.
 * Space Complexity: O(1) - Uses a constant amount of memory for variables.
 */
#include <iostream> // Required for standard input/output operations (cin, cout)
using namespace std; // Uses the standard namespace to avoid prefixing std::

int main() {
  // k: the cost of the first banana.
  // n: the amount of money the soldier has.
  // w: the number of bananas the soldier wants to buy.
  int k, n, w;
  // ossz: stores the total calculated cost for 'w' bananas. Using long long to prevent overflow.
  long long int ossz;
  
  // Reads the input values: k (first banana cost), n (money soldier has), w (number of bananas).
  cin >> k >> n >> w;
  
  /**
   * Block Logic: Calculates the total cost of 'w' bananas.
   * The cost forms an arithmetic progression: k, 2k, 3k, ..., wk.
   * The sum of this series is k * (1 + 2 + ... + w) = k * w * (w + 1) / 2.
   * The formula used in the code ((2 * k + (w - 1) * k) * w) / 2 is equivalent
   * to (k + wk) * w / 2 = k * (1 + w) * w / 2.
   * Invariant: 'ossz' holds the total money required to buy 'w' bananas.
   */
  ossz = ((2 * k + (w - 1) * k) * w) / 2;
  
  // Ternary operator to determine the amount to borrow.
  // If 'ossz - n' is less than or equal to 0, it means the soldier has enough money, so borrow 0.
  // Otherwise, the soldier needs to borrow 'ossz - n' dollars.
  ossz - n <= 0 ? cout << "0" : cout << ossz - n;

  return 0; // Indicates successful program execution.
}

