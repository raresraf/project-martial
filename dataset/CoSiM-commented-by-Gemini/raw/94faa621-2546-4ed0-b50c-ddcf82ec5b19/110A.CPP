/**
 * @file 110A.CPP
 * @brief Determines if a number is a "Nearly Lucky Number".
 *
 * This program solves the "Nearly Lucky Number" problem from Codeforces.
 * A number is considered nearly lucky if the count of its 'lucky digits' (4 and 7)
 * is itself a lucky number (i.e., the count is 4 or 7).
 *
 * @algorithm Digit Counting
 * @complexity
 * - Time: O(log10(N)), where N is the input number. The time complexity is
 *   proportional to the number of digits in N.
 * - Space: O(1).
 */
#include <bits/stdc++.h> // Includes iostream and other standard libraries.

int main() {
  // n: the input number.
  // c: counter for the number of lucky digits (4s and 7s).
  // g: stores the current digit being examined.
  long long int n, c = 0, g;
  std::cin >> n;

  /**
   * @brief Block Logic: Iterates through the digits of the input number to count lucky digits.
   * Invariant: After each iteration, `c` holds the count of lucky digits from the
   *            least significant digits of the original number that have been processed.
   *            `n` is reduced by one decimal place.
   */
  while (n > 0) {
    g = n % 10; // Extract the last digit.
    n /= 10;    // Remove the last digit.
    
    // Pre-condition: Check if the extracted digit is a lucky digit.
    if (g == 7 || g == 4)
      c++;
  }

  /**
   * @brief Block Logic: Checks if the count of lucky digits is itself a lucky number.
   * The final output is determined by whether the count `c` is 4 or 7.
   */
  std::cout << ((c == 7 || c == 4) ? "YES" : "NO");
}