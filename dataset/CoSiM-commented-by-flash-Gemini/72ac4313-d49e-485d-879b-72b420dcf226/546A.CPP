/**
 * @file 546A.CPP
 * @brief Solution to Codeforces Problem 546A: "Soldier and Bananas".
 * Calculates the amount of money a soldier needs to borrow to buy a certain number of bananas.
 * The cost of each banana increases incrementally (1st banana costs k, 2nd costs 2k, ..., w-th costs w*k).
 * Algorithm: Calculates the total cost of 'w' bananas using the sum of an arithmetic series.
 * Then, it determines if the soldier needs to borrow money by comparing the total cost
 * with the money they already possess.
 * Time Complexity: O(1) as it involves a fixed number of arithmetic operations.
 * Space Complexity: O(1) as only a few variables are used.
 */
// 28.01.19 @odnaks // Original author/date comment.
#include <iostream> // Required for input/output operations (std::cin, std::cout).

using namespace std;

/**
 * @brief Main function to solve the "Soldier and Bananas" problem.
 * Reads the cost of the first banana (k), initial money (n), and desired bananas (w).
 * Computes the total cost and the amount to borrow, then prints the result.
 */
int main() {
  int k; // Cost of the first banana.
  int n; // Amount of money the soldier has initially.
  int w; // Number of bananas the soldier wants to buy.
  int d; // Variable to store the total cost, and later the amount to borrow.

  // Reads the input values: cost of first banana (k), initial money (n), number of bananas to buy (w).
  cin >> k >> n >> w;
  
  // Calculates the total cost of 'w' bananas.
  // The cost of the i-th banana is i*k.
  // Total cost = k*1 + k*2 + ... + k*w = k * (1 + 2 + ... + w).
  // The sum 1 + 2 + ... + w is given by the formula w*(w+1)/2.
  // So, total cost = k * w * (w + 1) / 2.
  // The expression (k + w * k) * w / 2 simplifies to k * (1 + w) * w / 2,
  // which is mathematically equivalent to k * w * (w + 1) / 2.
  d = (k + w * k) * w / 2;
  
  /**
   * Block Logic: Determines the amount of money the soldier needs to borrow.
   * If the total calculated cost ('d') is greater than the money the soldier has ('n'),
   * the difference is the amount to borrow. Otherwise, no money needs to be borrowed.
   */
  if (d > n) {
    d = d - n; // Calculate the amount to borrow.
  } else {
    d = 0;     // No money needs to be borrowed.
  }
  
  cout << d; // Prints the final amount to borrow.
  return (0);
}