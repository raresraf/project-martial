/**
 * @file 546A.CPP
 * @brief Solution to Codeforces Problem 546A: "Soldier and Bananas".
 * Calculates the amount of money a soldier needs to borrow to buy a certain number of bananas.
 * The cost of each banana increases incrementally (1st banana costs k, 2nd costs 2k, ..., w-th costs w*k).
 * Algorithm: Calculate the total cost of 'w' bananas using the sum of an arithmetic series,
 * then determine the difference between the total cost and the soldier's initial money.
 * Time Complexity: O(1) as it involves a fixed number of arithmetic operations.
 * Space Complexity: O(1) as only a few variables are used.
 */

#include <iostream> // Required for input/output operations (std::cin, std::cout, std::endl).

/*
 * Original Contest Metadata:
 * Contest: Code Forces Round #304 (Task 546-A)
 * URL: http://codeforces.ru/contest/546/problem/A
 */

/**
 * @brief Main function to solve the "Soldier and Bananas" problem.
 * Reads the cost of the first banana (k), initial money (n), and desired bananas (w).
 * Computes the total cost, the amount to borrow, and prints the result.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 */
int main(int argc, char **argv) {
  int k, n, w, r; // 'k': cost of 1st banana, 'n': initial money, 'w': number of bananas, 'r': required money to borrow.
  
  // Reads the input values: cost of first banana (k), initial money (n), number of bananas to buy (w).
  std::cin >> k >> n >> w;

  // Calculates the total cost of 'w' bananas and then the amount needed to borrow ('r').
  // The sum of costs for 'w' bananas is k + 2k + ... + w*k = k * (1 + 2 + ... + w).
  // The sum 1 + 2 + ... + w is w*(w+1)/2.
  // So, total_cost = k * w * (w + 1) / 2.
  // 'r' is the total cost minus the money the soldier already has.
  r = (k * (w * (w + 1) / 2)) - n;

  // Prints the amount to borrow.
  // If 'r' is negative (soldier has enough or more money), prints 0.
  // Otherwise, prints 'r' (the positive amount to borrow).
  std::cout << ((r >= 0) ? r : 0) << std::endl;
  return 0;
}
