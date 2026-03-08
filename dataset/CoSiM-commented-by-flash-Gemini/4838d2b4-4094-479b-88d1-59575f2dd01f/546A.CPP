/**
 * @4838d2b4-4094-479b-88d1-59575f2dd01f/546A.CPP
 * @brief This program calculates the total cost of purchasing a series of items
 *        with increasing prices and determines the amount of money needed or remaining.
 * Algorithm: Arithmetic series sum for total cost, followed by a simple subtraction and max operation.
 * Time Complexity: O(1) as it involves a fixed number of arithmetic operations.
 * Space Complexity: O(1) for storing a few integer variables.
 */
#include <algorithm>
#include <ctype.h>
#include <iostream>
#include <map>
#include <math.h>
#include <set>
#include <stdio.h>
#include <string>

using namespace std;

int main() {
  // Declare variables:
  // k: The cost of the first banana.
  // n: The initial amount of money the soldier has.
  // w: The number of bananas the soldier wants to buy.
  int k, n, w;
  // Functional Utility: Read the input values for k, n, and w.
  cin >> k >> n >> w;
  // Block Logic: Calculate the total cost of 'w' bananas.
  // The cost of the i-th banana is i * k.
  // The total cost is k * (1 + 2 + ... + w) = k * w * (w + 1) / 2 (sum of an arithmetic series).
  int total = k * w * (w + 1) / 2;
  // Functional Utility: Calculate the amount of money the soldier needs to borrow.
  // If total cost is less than or equal to initial money, 0 is needed.
  // Otherwise, it's the difference between total cost and initial money.
  cout << max(total - n, 0);

  return 0;
}