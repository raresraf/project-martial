/**
 * @file 110A.CPP
 * @brief Solution for Codeforces Problem 110A: Nearly Lucky Number.
 * @details This program determines if a given integer 'n' is a "nearly lucky number".
 * A number is considered nearly lucky if the count of its digits '4' and '7' is itself
 * a lucky number (i.e., exactly 4 or exactly 7). The program iteratively extracts digits,
 * counts occurrences of 4s and 7s, and then checks if this count is 4 or 7.
 * @algorithm Digit extraction and counting.
 * Time Complexity: $O(\log_{10} N)$ where N is the input number, as it iterates through its digits.
 * Space Complexity: $O(1)$ to store a few variables.
 */

#include <iostream>
using namespace std;

/**
 * @file 110A.CPP
 * @brief Solution for Codeforces Problem 110A: Nearly Lucky Number.
 * @details This program determines if a given integer 'n' is a "nearly lucky number".
 * A number is considered nearly lucky if the count of its digits '4' and '7' is itself
 * a lucky number (i.e., exactly 4 or exactly 7). The program iteratively extracts digits,
 * counts occurrences of 4s and 7s, and then checks if this count is 4 or 7.
 * @algorithm Digit extraction and counting.
 * Time Complexity: $O(\log_{10} N)$ where N is the input number, as it iterates through its digits.
 * Space Complexity: $O(1)$ to store a few variables.
 */

#include <iostream> // Required for input/output operations (std::cin, std::cout).
using namespace std; // Uses the standard namespace.

int main() {
  /**
   * @brief Main function: orchestrates the "nearly lucky number" determination.
   */
  long long x; // @var x: The input number to check.
  cin >> x;    // Reads the input number 'x'.

  int c = 0;   // @var c: Counter for the occurrences of lucky digits ('4' or '7').

  /**
   * @brief Extracts digits from `x` and counts occurrences of '4' or '7'.
   * @block_logic Repeatedly takes the last digit of `x`, checks if it's '4' or '7', and removes it from `x`.
   * @invariant `x` is progressively reduced, and `c` accumulates the count of lucky digits.
   */
  while (x) { // Loop continues as long as `x` is not zero.
    int d = x % 10; // @var d: Stores the last digit of `x`.
    x = x / 10;     // Removes the last digit from `x`.
    if (d == 4 || d == 7) // Checks if the current digit is 4 or 7.
      c++;              // Increments the lucky digit counter.
  }

  /**
   * @brief Determines if the count of lucky digits (`c`) is itself a lucky number (4 or 7).
   * @block_logic Checks the final `c` value against the lucky numbers 4 and 7.
   * @invariant Prints "YES" if `c` is 4 or 7, otherwise "NO".
   */
  if (c == 4 || c == 7)
    cout << "YES"; // Output "YES" if the count of lucky digits is lucky.
  else
    cout << "NO";  // Output "NO" otherwise.

  return 0; // Indicates successful execution.
}
