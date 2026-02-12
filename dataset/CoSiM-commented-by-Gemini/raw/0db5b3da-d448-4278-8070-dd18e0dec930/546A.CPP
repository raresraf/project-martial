/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem using a recursive approach.
 * @details The program calculates the total cost for a soldier buying `w` bananas, where the
 * `i`-th banana costs `i*k`. It then determines the amount of money the soldier needs to borrow,
 * given their initial amount `n`. This implementation is unique in its use of recursion
 * to calculate the total cost.
 *
 * @algorithm Recursive summation of an arithmetic series.
 * @complexity
 * Time complexity: O(W), where W is the number of bananas, due to the W recursive calls.
 * Space complexity: O(W) due to the recursion depth, which can lead to a stack overflow for large W.
 * This is less efficient than an iterative or a direct formula-based O(1) solution.
 */
#include <iostream>

using std::cin;
using std::cout;
using std::endl;

// Function prototype for the recursive cost calculation.
int cost(int, int);

int main() {
  // k: the cost of the first banana.
  // n: the initial money the soldier has.
  // w: the number of bananas to buy.
  int k, n, w;
  cin >> k >> n >> w;
  
  // Calculate the total cost recursively and determine the amount to borrow.
  int borrowNeeded = cost(k, w) - n;

  // If the soldier has enough money, the amount to borrow is 0.
  if (borrowNeeded < 0) {
    cout << "0";
  } else {
    cout << borrowNeeded;
  }

  return 0;
}

/**
 * @brief Calculates the total cost of `i` bananas recursively.
 * @param k The base cost of the first banana.
 * @param i The number of bananas to calculate the cost for.
 * @return The total cost for `i` bananas.
 */
int cost(int k, int i) {
  // Base case: The cost of one banana is just k.
  if (i == 1) {
    return k;
  }

  // Recursive step: The cost of `i` bananas is the cost of the i-th banana (`i*k`)
  // plus the total cost of the first `i-1` bananas.
  return i * k + cost(k, i - 1);
}