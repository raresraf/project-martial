/**
 * @file 546A.CPP
 * @brief This competitive programming solution addresses the "Soldier and Bananas" problem.
 *        A soldier wants to buy a certain number of bananas. The cost of the first banana is `cost`,
 *        the second is `2 * cost`, the third is `3 * cost`, and so on. The soldier has `mon` dollars
 *        and needs to buy `need` bananas. The task is to calculate how much money the soldier needs
 *        to borrow from his friend to buy all the bananas, or 0 if he has enough.
 * Algorithm: Iterative Summation.
 *            1. Read the base cost of a banana (`cost`), the money the soldier has (`mon`),
 *               and the number of bananas he wants to buy (`need`).
 *            2. Calculate the total cost of buying `need` bananas by summing `i * cost` for `i` from 1 to `need`.
 *            3. Compare the total cost (`tot`) with the money the soldier has (`mon`).
 *            4. If `tot` is less than or equal to `mon`, the soldier needs to borrow 0.
 *            5. Otherwise, the soldier needs to borrow `tot - mon`.
 * Time Complexity: O(N), where N is the `need` (number of bananas). This is due to the `for` loop
 *                  iterating `need` times to calculate the total cost.
 * Space Complexity: O(1), as only a few fixed-size integer variables are used to store inputs and intermediate calculations.
 */
#include <bits/stdc++.h> // Includes various standard library headers, such as iostream.
using namespace std;

/**
 * @brief Main function of the program. Calculates the total cost of bananas and the amount to borrow.
 * Functional Utility: Serves as the entry point for the "Soldier and Bananas" problem.
 */
int main() {
  long long int cost; // Cost of the first banana.
  long long int mon;  // Money the soldier has.
  long long int need; // Number of bananas the soldier wants to buy.
  long long int tot = 0; // Variable to accumulate the total cost of bananas.
  
  cin >> cost >> mon >> need; // Reads the input values.

  // Block Logic: Calculates the total cost of buying `need` bananas.
  // The cost of the i-th banana is `i * cost`.
  // Invariant: `tot` accumulates the sum of costs for bananas from 1 to `i`.
  for (int i = 1; i <= need; i++) {
    tot += i * cost; // Adds the cost of the current banana to the total.
  }
  
  // Block Logic: Determines if the soldier needs to borrow money and calculates the amount.
  if (tot <= mon) // If the total cost is less than or equal to the money the soldier has.
    cout << "0" << endl; // Prints "0" as no money needs to be borrowed.
  else
    cout << tot - mon << endl; // Prints the amount to borrow (total cost - money he has).

  return 0; // Indicates successful program termination.
}
