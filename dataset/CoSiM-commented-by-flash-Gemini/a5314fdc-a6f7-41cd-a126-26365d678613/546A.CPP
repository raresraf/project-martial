/**
 * @file 546A.CPP
 * @brief Solution to the "Soldier and Bananas" problem.
 *
 * This program calculates the amount of money a soldier needs to borrow to buy 'w' bananas.
 * The cost of the first banana is 'k', the second is '2k', and so on, up to 'wk'.
 * Algorithm: Summation of an arithmetic progression. The total cost is k * (1 + 2 + ... + w).
 *            The sum (1 + 2 + ... + w) can be calculated as w * (w + 1) / 2.
 *            The amount to borrow is the total cost minus the money the soldier already has ('n'),
 *            or 0 if the soldier has enough money.
 * Time Complexity: O(W) where W is the number of bananas, due to the explicit loop for summation.
 *                  (Note: This could be optimized to O(1) using the arithmetic series formula).
 * Space Complexity: O(1) - constant space used for variables.
 */

#include <iostream> // Required for standard input/output operations (std::cin, std::cout).

using namespace std; // Use the standard namespace for convenience.

/**
 * @brief Main function to calculate the borrowed amount for bananas.
 *
 * Reads the cost of the first banana, initial money, and number of bananas.
 * Calculates the total cost and the amount to borrow, then prints the result.
 * @return 0 on successful execution.
 */
int main() {
  long long k; // Cost of the first banana.
  long long d; // Money the soldier has (corresponds to 'n' in problem statement).
  long long n; // Number of bananas to buy (corresponds to 'w' in problem statement).
  long long b = 0; // Accumulated total cost of bananas.

  // Block Logic: Read the input values for banana cost ('k'), money soldier has ('d'),
  //              and number of bananas to buy ('n').
  // Pre-condition: User provides three positive integers.
  // Invariant: k, d, and n hold the respective input values.
  cin >> k >> d >> n;

  // Block Logic: Calculate the total cost of 'n' bananas using a loop.
  // The cost increases linearly: 1*k, 2*k, ..., n*k.
  // This is an arithmetic series sum, which can also be calculated as k * n * (n + 1) / 2.
  // Pre-condition: 'n' is a positive integer.
  // Invariant: 'b' accumulates the sum of costs for 'i' bananas.
  for (int i = 1; i <= n; i++)
    b += i * k;

  // Block Logic: Calculate the amount of money to borrow and print the result.
  // If the total cost 'b' is greater than the money the soldier has 'd',
  // the soldier needs to borrow 'b - d'. Otherwise, no money needs to be borrowed (0).
  // Post-condition: The non-negative amount of money to borrow is printed to standard output.
  cout << (b - d > 0 ? b - d : 0);
  return 0; // Indicate successful execution.
}