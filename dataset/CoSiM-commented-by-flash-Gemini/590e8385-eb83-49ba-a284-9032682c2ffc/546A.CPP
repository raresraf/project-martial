/**
 * @590e8385-eb83-49ba-a284-9032682c2ffc/546A.CPP
 * @brief This program calculates the amount of money a soldier needs to borrow.
 * The cost of buying bananas increases linearly: the first costs 'k', the second '2k',
 * and so on. Given 'w' bananas to buy and 'n' dollars the soldier already has,
 * the program determines the deficit.
 *
 * Algorithm: Iterative summation to calculate total cost, followed by a subtraction and conditional output.
 * Time Complexity: O(W) where W is the number of bananas.
 * Space Complexity: O(1) for storing variables.
 */

#include <bits/stdc++.h> // A common header in competitive programming; includes many standard libraries.

using namespace std; // Uses the standard namespace to avoid prefixing std::.
#define ll long long // Defines 'll' as a shorthand for long long, used for 'n' and 'sum' to prevent overflow.

int main() {
  int k, w; // Declares 'k' for the cost of the first banana and 'w' for the number of bananas.
  ll n, sum = 0; // Declares 'n' for initial money and 'sum' (total cost) initialized to 0.
                  // 'll' is used for 'n' and 'sum' to handle potentially large values.

  cin >> k >> n >> w; // Reads 'k', 'n', and 'w' from standard input.
  
  /**
   * Block Logic: Calculates the total cost of buying 'w' bananas.
   * The cost of the i-th banana is 'i * k'.
   * Invariant: 'sum' accurately accumulates the total cost after each banana is accounted for.
   */
  for (int i = 1; i <= w; i++) { // Loop from the 1st banana to the w-th banana.
    sum += (ll)k * i;           // Inline: Calculates the cost of the current banana (i * k) and adds it to 'sum'.
                                // (ll)k ensures the multiplication is performed using long long to prevent overflow.
  }
  
  /**
   * Block Logic: Determines the amount of money to borrow.
   * Pre-condition: 'sum' contains the total cost of bananas and 'n' is the available money.
   * Invariant: The program outputs the required borrowing amount or 0 if no borrowing is needed.
   */
  if (sum - n > 0) // If the total cost exceeds the available money.
    cout << sum - n; // Prints the positive difference (amount to borrow).
  else
    cout << "0"; // Prints 0 if no money needs to be borrowed (sum <= n).
  return 0;      // Indicates successful program termination.
}
