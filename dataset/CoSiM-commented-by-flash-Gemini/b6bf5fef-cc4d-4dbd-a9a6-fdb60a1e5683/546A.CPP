/**
 * @file 546A.CPP
 * @brief Solution to a competitive programming problem involving calculating money needed.
 * This program calculates the total cost of purchasing 'w' items, where the cost
 * of each item increases arithmetically. It then determines if the user has
 * enough money 'n' to cover the total cost, printing the deficit if any, or 0 if no deficit.
 * This version uses the sum of an arithmetic series for optimized calculation.
 * Algorithm: Arithmetic progression sum formula with conditional output.
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */
// http://ideone.com/BQzJDe
// 546A bananas

#include <iostream>

typedef long long ll;

using namespace std;

/**
 * @brief Main function where the program execution begins.
 * Handles input for initial cost, available money, and number of items,
 * then calculates and displays the financial outcome.
 */
int main() {

  ll n, w, k, res;

  // Reads the initial cost 'k', available money 'n', and number of items 'w'.
  cin >> k >> n >> w;

  // Calculates the total cost 'res' using the sum of an arithmetic series formula:
  // S = (w * (w + 1)) / 2 * k, where 'w' is the number of terms, '1' is the first term
  // in the series of multipliers, and 'k' is the common difference/base cost.
  res = ((w * (w + 1)) / 2) * k;

  /**
   * Block Logic: Determines if there is a deficit and prints the result.
   * Pre-condition: 'res' holds the total calculated cost, 'n' holds available money.
   * Invariant: The output represents the money needed or 0 if sufficient funds exist.
   */
  if (res <= n)
    // If available money 'n' is more than or equal to total cost 'res', print 0.
    cout << "0";
  else
    // Otherwise, print the deficit (total cost - available money).
    cout << res - n;

  return 0;
}
