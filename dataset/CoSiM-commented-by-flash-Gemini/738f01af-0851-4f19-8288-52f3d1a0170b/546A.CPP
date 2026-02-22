/**
 * @file 546A.CPP
 * @brief Solution for Codeforces Problem 546A: Soldier and Bananas.
 * @details This program calculates the amount of money a soldier needs to borrow to buy 'n' bananas,
 * given that the first banana costs 'k' dollars, the second '2k' dollars, and so on (the i-th banana
 * costs `i*k` dollars), and the soldier currently has 'w' dollars. It calculates the total cost
 * of all bananas and then determines the difference if the total cost exceeds the available money.
 * @algorithm Arithmetic series summation and conditional calculation.
 * Time Complexity: $O(1)$ as it involves a simple arithmetic formula.
 * Space Complexity: $O(1)$ to store a few variables.
 */
#include <bits/stdc++.h>

using namespace std;

/**
 * @file 546A.CPP
 * @brief Solution for Codeforces Problem 546A: Soldier and Bananas.
 * @details This program calculates the amount of money a soldier needs to borrow to buy 'n' bananas,
 * given that the first banana costs 'k' dollars, the second '2k' dollars, and so on (the i-th banana
 * costs `i*k` dollars), and the soldier currently has 'w' dollars. It calculates the total cost
 * of all bananas and then determines the difference if the total cost exceeds the available money.
 * @algorithm Arithmetic series summation and conditional calculation.
 * Time Complexity: $O(1)$ as it involves a simple arithmetic formula.
 * Space Complexity: $O(1)$ to store a few variables.
 */
#include <bits/stdc++.h> // Standard C++ library includes (e.g., iostream for cin/cout).

using namespace std; // Using the standard namespace to avoid prefixing std::.

int main() {
  /**
   * @brief Main function: orchestrates the banana cost calculation.
   */
  long long k, // @var k: Cost of the first banana.
            w, // @var w: Amount of money the soldier currently has.
            n; // @var n: Number of bananas the soldier wants to buy.

  // Functional Utility: Reads the input values k, w, and n.
  cin >> k >> w >> n;

  /**
   * @brief Calculates the total cost of buying 'n' bananas.
   * @details The cost follows an arithmetic progression: k, 2k, 3k, ..., nk.
   * The sum of this series is k * (1 + 2 + ... + n) = k * n * (n + 1) / 2.
   */
  long long ans = n * (n + 1) / 2 * k; // @var ans: Stores the total cost of 'n' bananas.

  /**
   * @brief Determines the amount of money to borrow.
   * @block_logic Compares the total cost (`ans`) with the available money (`w`).
   * @invariant If `ans` is less than or equal to `w`, no money needs to be borrowed (0).
   *            Otherwise, the difference (`ans - w`) is the amount to borrow.
   */
  if (ans <= w)
    cout << 0 << endl; // If total cost is less than or equal to available money, print 0.
  else
    cout << ans - w << endl; // Otherwise, print the amount that needs to be borrowed.

  return 0; // Indicates successful execution.
}
