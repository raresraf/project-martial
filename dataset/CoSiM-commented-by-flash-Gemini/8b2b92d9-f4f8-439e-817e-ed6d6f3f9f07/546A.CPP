/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem from Codeforces (Problem 546A).
 *        This program calculates the total cost of buying 'w' bananas, where the
 *        'i'-th banana costs 'i * k'. It then determines how much money a soldier
 *        needs to borrow if he initially has 'n' dollars.
 * Algorithm: Simple Summation
 * Time Complexity: O(W), where W is the number of bananas.
 * Space Complexity: O(1).
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to calculate the total cost of bananas and determine
 *        the amount of money to borrow.
 * @return 0 if the program executes successfully.
 */
int main() {
  // Declare variables:
  // k: cost of the first banana.
  // n: initial money the soldier has.
  // w: number of bananas the soldier wants to buy.
  int k, n, w;
  // Read k, n, and w from standard input.
  cin >> k >> n >> w;
  // Initialize totalCost to 0.
  long long totalCost = 0; // Use long long to prevent potential overflow for totalCost.
  
  /**
   * Block Logic: Calculates the total cost of purchasing 'w' bananas.
   * Invariant: For each 'i'-th banana, its cost is 'i * k'.
   */
  for (int i = 1; i <= w; i++)
    // Accumulate the cost of the current banana.
    totalCost += (long long)i * k; // Cast to long long to ensure multiplication is done with wider type.
  
  /**
   * Block Logic: Determines the amount of money the soldier needs to borrow.
   * Pre-condition: 'totalCost' holds the sum of all banana costs.
   * Post-condition: Prints the amount to borrow if 'totalCost' exceeds 'n', otherwise prints 0.
   */
  if (totalCost > n)
    // If the total cost is greater than the money the soldier has, print the difference.
    cout << totalCost - n;
  else
    // Otherwise, the soldier has enough money or more, so print 0.
    cout << 0;
  
  return 0; // Indicate successful execution.
}
