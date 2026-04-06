/**
 * @file 339A.CPP
 * @brief Solution for Codeforces Problem 339A - "Helpful Maths".
 * This program takes a sum of single-digit numbers (1, 2, or 3) as a string
 * (e.g., "3+1+2") and reorders the numbers in non-decreasing order, printing
 * the resulting sum string (e.g., "1+2+3").
 *
 * @problem_statement
 * Given a string representing a sum of numbers (1, 2, or 3) separated by '+',
 * sort these numbers in ascending order and print the sum.
 *
 * @algorithm
 * 1. Read the input sum as a string.
 * 2. Sort the characters of the entire string. Since the numbers are single digits
 *    and '+' comes before digits lexicographically, sorting the characters
 *    will group '+' signs together and sort the digits.
 * 3. Iterate through the sorted string:
 *    a. Skip any '+' characters.
 *    b. For each digit character encountered, print a '+' before it if it's not
 *       the very first digit being printed.
 *    c. Print the digit.
 *
 * Time Complexity: O(L log L) due to sorting the string, where L is the length of the string.
 * Space Complexity: O(L) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  /**
   * @brief Main function to read, sort, and print the "Helpful Maths" sum string.
   * Reads the input string, sorts its characters to arrange the numbers, and then
   * prints the numbers back with '+' signs in their sorted order.
   * @return 0 on successful execution.
   */
  string str; ///< @var str The input string containing the sum (e.g., "3+1+2").
  int ck = 0; ///< @var ck Control flag to manage printing '+' signs. `0` indicates the first number, `1` indicates subsequent numbers.

  cin >> str; // Read the input sum string.

  /**
   * @brief Sorts the characters of the input string.
   * @details Due to ASCII/Unicode ordering, characters '1', '2', '3' come after
   * '+' character. Sorting the string directly arranges the numbers in ascending
   * order while grouping '+' characters together.
   * @post `str` contains the same characters as before, but sorted lexicographically.
   */
  sort(str.begin(), str.end());

  /**
   * @brief Iterates through the sorted string to print the numbers with '+' separators.
   * @invariant `i` progresses through the sorted characters of `str`. `ck` ensures
   * '+' is printed only between numbers.
   * @post The sorted numbers are printed to standard output, separated by '+'.
   */
  for (int i = 0; i < str.size(); i++) {
    /**
     * @brief Skips '+' characters as they are separators and not numbers to be printed.
     */
    if (str[i] == '+')
      continue;
    /**
     * @brief Prints a '+' separator before subsequent numbers.
     * @details `ck != 0` ensures that a '+' is not printed before the very first number.
     */
    if (ck != 0)
      cout << "+";
    cout << str[i]; // Print the current digit.
    ck = 1; // Set flag to indicate that at least one number has been printed.
  }
  cout << endl; // Print a newline at the end of the output.
  return 0;
}
