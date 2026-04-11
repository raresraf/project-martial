/**
 * @file 546A.CPP
 * @brief Solution to a competitive programming problem involving calculating money needed.
 * This program calculates the total cost of purchasing 'w' items, where the cost
 * of each item increases arithmetically. It then determines if the user has
 * enough money 'n' to cover the total cost, printing the deficit if any, or 0 if no deficit.
 * Algorithm: Arithmetic progression sum calculation with conditional output.
 * Time Complexity: O(W) where W is the number of items.
 * Space Complexity: O(1)
 */
#include <iostream>
using namespace std;
typedef long long ll;

/**
 * @brief Main function where the program execution begins.
 * Handles input for initial cost, available money, and number of items,
 * then calculates and displays the financial outcome.
 */
int main() {
  ll k, n, w;
  // Reads the initial cost 'k', available money 'n', and number of items 'w'.
  cin >> k >> n >> w;
  ll dollar = 0;
  /**
   * Block Logic: Calculates the total cost of 'w' items.
   * Invariant: 'dollar' accumulates the total cost, 'i' represents the current item number.
   * Pre-condition: 'k', 'n', 'w' are non-negative long long integers.
   */
  for (int i = 1; i <= w; i++) {
    // Adds the cost of the current item (k * i) to the total 'dollar'.
    dollar += k * i;
  }
  /**
   * Block Logic: Determines if there is a deficit and prints the result.
   * Pre-condition: 'dollar' holds the total cost, 'n' holds available money.
   * Invariant: The output represents the money needed or 0 if sufficient funds exist.
   */
  if ((dollar - n) < 0) {
    // If available money 'n' is more than or equal to total cost 'dollar', print 0.
    cout << "0";
  } else
    // Otherwise, print the deficit (total cost - available money).
    cout << dollar - n;
  return 0;
}