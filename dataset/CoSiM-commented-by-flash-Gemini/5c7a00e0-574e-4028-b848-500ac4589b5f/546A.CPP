/**
 * @5c7a00e0-574e-4028-b848-500ac4589b5f/546A.CPP
 * @brief This program calculates the amount of money a soldier needs to borrow.
 * The cost of buying bananas increases linearly: the first costs 'k', the second '2k',
 * and so on, with the 'i'-th banana costing 'i * k' dollars. Given 'w' bananas to buy
 * and 'n' dollars the soldier already has, the program efficiently determines the deficit.
 *
 * Algorithm: Direct mathematical formula for sum of an arithmetic series, followed by
 * subtraction and conditional output.
 * Time Complexity: O(1) as it involves a constant number of arithmetic operations.
 * Space Complexity: O(1) for storing variables.
 */

#include <iostream> // For standard input/output operations (cin, cout).
using namespace std; // Uses the standard namespace.

int main() {
  // Declares 'k' (cost of first banana), 'n' (initial money), 'w' (number of bananas),
  // and 'ans' (amount to borrow), all as long long int to prevent overflow.
  long long int k, n, w, ans; 
  cin >> k >> n >> w; // Reads 'k', 'n', and 'w' from standard input.
  
  // Block Logic: Calculates the amount of money to borrow using a direct formula.
  // The total cost of 'w' bananas is k * (1 + 2 + ... + w).
  // The sum 1 + 2 + ... + w is calculated as w * (w + 1) / 2.
  // The expression (k + w * k) * w / 2 simplifies to k * (1 + w) * w / 2,
  // which is equivalent to k * (w * (w + 1) / 2).
  // Then, 'n' (money owned) is subtracted from the total cost to find the deficit.
  ans = (k + w * k) * w / 2 - n; 
  
  // Block Logic: Ensures that the amount to borrow is not negative.
  // If the calculated deficit 'ans' is negative (meaning enough or more money was available),
  // it is set to 0.
  if (ans < 0)
    ans = 0;
  cout << ans << endl; // Prints the final amount to borrow, followed by a newline.
  return 0;            // Indicates successful program termination.
}
