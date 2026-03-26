/**
 * @file 546A.CPP
 * @brief This program calculates the total cost of purchasing a certain number of bananas,
 * where the cost of each successive banana increases linearly. It then determines
 * if a soldier has enough money to buy all bananas and, if not, calculates the deficit.
 * Algorithm: Linear iteration to sum costs, followed by a conditional check.
 * Time Complexity: O(W) where W is the number of bananas.
 * Space Complexity: O(1) for storing a few variables.
 */

#include <iostream> // Required for input/output operations (cin, cout).
using namespace std;

/**
 * @brief Main function to calculate banana purchase cost and money deficit.
 * It reads the initial cost of one banana, the soldier's available dollars,
 * and the number of bananas desired. It then calculates the total cost
 * and outputs the amount of money the soldier needs to borrow (or 0 if sufficient).
 * @return 0 on successful execution.
 */
int main() {
  // @var k: Long integer representing the cost of the first banana.
  // @var n: Long integer representing the number of dollars the soldier has.
  // @var w: Long integer representing the number of bananas the soldier wants to buy.
  long k, n, w;
  // Read the cost of the first banana (k), the soldier's money (n), and the number of bananas (w).
  cin >> k >> n >> w;
  // @var price: Long integer to accumulate the total cost of all bananas. Initialized to 0.
  long price = 0;
  /**
   * @brief Loop to calculate the total cost of buying 'w' bananas.
   * The i-th banana costs i * k dollars.
   * Invariant: At the start of each iteration, 'price' holds the accumulated cost
   * of bananas from 1 to (i-1).
   * Pre-condition: 'k' is the base cost, 'w' is the target number of bananas.
   */
  for (int i = 1; i <= w; i++) {
    price += (i * k); // Add the cost of the current (i-th) banana to the total price.
  }
  /**
   * @brief Conditional logic to determine if the soldier has enough money.
   * If the soldier's money ('n') is greater than or equal to the total price,
   * no money needs to be borrowed. Otherwise, calculate the deficit.
   */
  if (n >= price) {
    cout << "0" << endl; // Output 0, indicating no money needs to be borrowed.
  } else {
    cout << price - n << endl; // Output the difference, which is the amount of money to borrow.
  }
  return 0;
}
