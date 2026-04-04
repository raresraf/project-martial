/**
 * @file 110A.CPP
 * @brief Determines if a number is "almost lucky".
 *
 * This program reads a single positive integer. It then counts how many digits
 * in the number are 'lucky digits' (defined as 4 or 7). Finally, it checks
 * if this count of lucky digits is itself a lucky number (i.e., if the count is 4 or 7).
 * Based on this final check, it outputs "YES" or "NO".
 *
 * Algorithm: Digit extraction and counting. The number is repeatedly divided by 10
 *            to extract its digits.
 * Time Complexity: O(log N), where N is the input number, as the loop runs
 *                  proportionally to the number of digits in N.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * @brief Main function: Checks if an input number is "almost lucky".
   * @return 0 if the program executes successfully.
   */
  long long num; // Declare a long long integer to handle potentially large input numbers.
  cin >> num;    // Read the input number.
  int count = 0; // Initialize a counter for lucky digits.
  /**
   * @brief Loop to extract digits and count occurrences of lucky digits (4 or 7).
   * @pre 'num' holds the current value of the number being processed.
   * @post 'count' holds the total number of lucky digits found in the original number.
   * @invariant 'num' is reduced by a factor of 10 in each iteration, effectively
   *            processing digits from right to left.
   */
  while (num != 0) {
    /**
     * @brief Checks if the last digit of 'num' is a lucky digit.
     * @pre 'num % 10' gives the last digit.
     */
    if (num % 10 == 4 || num % 10 == 7) {
      count += 1; // Increment count if the digit is 4 or 7.
    }
    num /= 10; // Remove the last digit by integer division.
  }
  /**
   * @brief Determines if the count of lucky digits is itself a lucky number (4 or 7).
   * @pre 'count' holds the total number of lucky digits.
   * @post Outputs "YES" or "NO" based on the condition.
   */
  if (count == 4 || count == 7) {
    cout << "YES" << endl; // Output "YES" if the count is 4 or 7.
  } else {
    cout << "NO" << endl; // Otherwise, output "NO".
  }
  return 0;
}
