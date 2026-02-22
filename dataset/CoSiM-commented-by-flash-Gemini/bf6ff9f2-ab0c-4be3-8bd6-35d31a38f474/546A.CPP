/**
 * @bf6ff9f2-ab0c-4be3-8bd6-35d31a38f474/546A.CPP
 * @brief Solution for a competitive programming problem involving calculating borrowed money for purchases.
 *
 * This program calculates the amount of money a soldier needs to borrow to buy a certain number of bananas.
 * The cost of bananas increases arithmetically: the first costs 'k', the second '2k', up to 'wk'.
 * The soldier has 'n' dollars initially. The task is to find the deficit, or 0 if no borrowing is needed.
 *
 * Algorithm:
 * 1. Read the base cost of a banana 'k', the initial money 'n', and the number of bananas 'w'.
 * 2. Calculate the total cost of buying 'w' bananas. The sum of costs (k + 2k + ... + wk)
 *    is equivalent to k * (1 + 2 + ... + w), which simplifies to k * (w * (w + 1) / 2).
 * 3. Calculate the difference between the total cost and the initial money.
 * 4. If this difference is negative (meaning the soldier has enough or extra money), the amount to borrow is 0.
 * 5. Otherwise, the difference is the amount that needs to be borrowed.
 *
 * Time Complexity: O(1) - constant time operations for input, calculation, and output.
 * Space Complexity: O(1) - uses a fixed amount of memory for variables.
 */

#include <iostream> // Functional Utility: Provides standard input/output streams (std::cin, std::cout).
using namespace std;

int main() {
  int k, n, w;
  // Functional Utility: Reads the base cost of one banana 'k', initial money 'n', and number of bananas 'w'.
  cin >> k >> n >> w;
  // Functional Utility: Variable to store the total calculated cost.
  int sum;
  // Functional Utility: Calculates the total cost using the sum of an arithmetic progression formula.
  // The sum of 1 to w is w * (w + 1) / 2. This is then multiplied by 'k'.
  sum = (w * (w + 1) / 2) * k;
  // Functional Utility: Calculates the amount of money that needs to be borrowed.
  int borrow = sum - n;
  // Conditional Logic: Checks if the calculated 'borrow' amount is negative (i.e., enough money or surplus).
  if (borrow < 0)
    // Functional Utility: If no borrowing is needed, prints 0.
    cout << "0" << endl;
  else
    // Functional Utility: Otherwise, prints the calculated amount to be borrowed.
    cout << borrow << endl;
  return 0; // Functional Utility: Indicates successful program execution.
}
