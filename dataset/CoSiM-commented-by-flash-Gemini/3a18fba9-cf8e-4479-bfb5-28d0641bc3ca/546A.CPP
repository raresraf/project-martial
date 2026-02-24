/**
 * @file 546A.CPP
 * @brief Solution for Codeforces Problem 546A - Soldier and Bananas.
 *
 * This program calculates the amount of money a soldier needs to borrow to buy
 * a certain number of bananas. The first banana costs 'k', the second '2k',
 * and so on. The soldier has 'n' money and wants to buy 'w' bananas.
 *
 * Algorithm: Sum of an arithmetic series. The total cost of 'w' bananas is
 * calculated using the formula for the sum of the first 'w' terms of an
 * arithmetic series where the first term is 'k' and the common difference is 'k'.
 * Total Cost = k * (1 + 2 + ... + w) = k * w * (w + 1) / 2.
 * The amount to borrow is then (Total Cost - n), if this value is positive.
 *
 * Time Complexity: O(1) - constant time operations.
 * Space Complexity: O(1) - constant space usage.
 *
 * SubmissionId: 17336820
 * ContestId: 546
 * Index: A
 * ProblemName: Soldier and Bananas
 * ProblemTags: implementation, math
 * ProgrammingLanguage: GNU C++11
 * Verdict: OK
 */

#include <bits/stdc++.h> // Includes various standard libraries.
typedef long long ll;    // Alias for long long integer.
typedef unsigned long long ull; // Alias for unsigned long long integer.
#define pi                                                                     \
  3.141592653589793238462643383279502884197169399375105820974944592307816406286 // Definition for PI (unused).
using namespace std; // Uses the standard namespace.

/**
 * @brief Checks if a given unsigned long long integer is prime.
 *
 * This is a standard primality test function, likely part of a competitive
 * programming template. It is not used in the main logic of this problem.
 *
 * @param n The number to check for primality.
 * @return True if 'n' is prime, False otherwise.
 */
bool isprime(ull n) {
  if (n <= 1)
    return false;
  if (n <= 3)
    return true;
  if (n % 2 == 0 || n % 3 == 0)
    return false;
  for (ull i = 5; i * i <= n; i = i + 6)
    if (n % i == 0 || n % (i + 2) == 0)
      return false;
  return true;
}

int main() {
  /**
   * @brief Main function to execute the Soldier and Bananas problem logic.
   *
   * Optimizes C++ standard stream I/O for faster execution.
   * Reads the first banana cost 'k', initial money 'n', and number of bananas 'w'.
   * Calculates the total cost of 'w' bananas and then determines the amount
   * the soldier needs to borrow. If he doesn't need to borrow, it prints 0.
   */
  // Functional Utility: Decouples C++ streams from standard C streams and unties cin from cout.
  // This optimizes high-throughput I/O often required for competitive programming.
  ios_base::sync_with_stdio(0);
  cin.tie(NULL);
  cout.tie(NULL);

  // Declare long long integers for 'k' (first banana cost), 'n' (initial money),
  // 'w' (number of bananas), and 'b' (borrowed amount).
  long long int k, n, w, b;
  // Read the input values: k, n, w.
  cin >> k >> n >> w;
  // Calculate the amount to borrow.
  // The total cost of 'w' bananas is k * (1 + 2 + ... + w) = k * w * (w + 1) / 2.
  // 'b' is the difference between total cost and initial money.
  b = (k * w * (w + 1) / 2) - n;
  // Conditional Logic: Determine if money needs to be borrowed.
  // If 'b' is negative, it means the soldier has enough money or more, so he needs to borrow 0.
  if (b < 0)
    cout << "0";
  else
    // Otherwise, print the calculated amount 'b' that needs to be borrowed.
    cout << b;
  return 0; // Indicate successful program execution.
}
