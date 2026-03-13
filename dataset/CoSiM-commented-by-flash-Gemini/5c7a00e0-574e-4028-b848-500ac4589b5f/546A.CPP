/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem from Codeforces (546A).
 *
 * @details This program calculates the amount of money a soldier needs to borrow to buy a
 * certain number of bananas. The cost of the bananas increases linearly: the first
 * banana costs 'k', the second '2k', ..., and the 'i'-th banana costs 'i * k'.
 * The program calculates the total cost and subtracts the soldier's initial money 'n'
 * to find the deficit.
 *
 * @section Algorithm
 * The total cost of buying 'w' bananas forms an arithmetic series:
 * `Cost = k + 2k + 3k + ... + wk`
 * This can be factored to `Cost = k * (1 + 2 + 3 + ... + w)`.
 * The sum of the first 'w' positive integers is `w * (w + 1) / 2`.
 * Therefore, the total cost is `k * w * (w + 1) / 2`.
 * The amount to borrow is the `Total Cost - n`. If this value is negative (meaning
 * the soldier has enough money), the amount to borrow is 0.
 *
 * @section Complexity
 * - Time Complexity: O(1), as the solution involves a fixed number of arithmetic operations.
 * - Space Complexity: O(1).
 */
#include <iostream>
using namespace std;

/**
 * @brief Main function to execute the solution logic.
 * @return 0 on successful execution.
 */
int main() {
  /**
   * @var k The cost of the first banana.
   * @var n The initial amount of money the soldier has.
   * @var w The total number of bananas the soldier wants to buy.
   * @var ans The final calculated amount of money to borrow.
   * Using 'long long int' prevents potential integer overflow for large inputs.
   */
  long long int k, n, w, ans; 
  cin >> k >> n >> w; 
  
  /**
   * @brief Block Logic: Calculates the total cost of 'w' bananas using the
   * formula for the sum of an arithmetic series, then determines the deficit.
   * The formula `(k + w * k) * w / 2` is equivalent to `k * w * (w + 1) / 2`.
   */
  ans = (k + w * k) * w / 2 - n; 
  
  /**
   * @brief Block Logic: Ensures the amount to borrow is non-negative.
   * If 'ans' is less than 0, it means the soldier has enough money, so the
   * amount to borrow is 0.
   */
  if (ans < 0)
    ans = 0;
  
  // Prints the final calculated amount to borrow.
  cout << ans << endl;
  return 0;
}