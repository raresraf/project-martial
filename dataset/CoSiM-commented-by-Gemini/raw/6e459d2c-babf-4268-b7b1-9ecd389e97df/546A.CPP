
#include <iostream>
using namespace std;

/**
 * @file 546A.CPP
 * @brief Solution to Codeforces problem 546A, "Soldier and Bananas".
 * @details The program calculates the amount of money a soldier needs to borrow
 * to buy `w` bananas, where the cost increases linearly.
 *
 * Algorithm:
 * The total cost forms an arithmetic progression: k, 2k, 3k, ..., wk.
 * This implementation uses the standard formula for the sum of an arithmetic
 * progression: `Sum = (num_terms / 2) * (2 * first_term + (num_terms - 1) * diff)`.
 * Here, `num_terms = w`, `first_term = k`, and `diff = k`. The code calculates
 * this sum directly. It then uses a ternary operator to determine the result:
 * if the soldier's initial money `n` is enough to cover the total cost `ossz`,
 * it prints 0; otherwise, it prints the difference `ossz - n`.
 *
 * Time Complexity: O(1), as it uses a direct mathematical formula.
 * Space Complexity: O(1).
 */
int main() {
  // k: base cost of a banana.
  // n: initial money the soldier has.
  // w: number of bananas to buy.
  int k, n, w;
  long long int ossz; // Will store the total cost.
  cin >> k >> n >> w;

  /**
   * @brief Calculate the total cost using the arithmetic progression sum formula.
   * `ossz = (w/2) * (2*k + (w-1)*k)`
   */
  ossz = ((2 * k + (w - 1) * k) * w) / 2;

  /**
   * @brief Print the amount to borrow using a ternary operator.
   * If `ossz - n` is zero or negative, the soldier has enough money.
   */
  ossz - n <= 0 ? cout << "0" : cout << ossz - n;

  return 0;
}
