/**
 * @file 110A.CPP
 * @brief Determines if an input number is a "Nearly Lucky Number".
 *        A number is nearly lucky if the count of its 'lucky digits' (4s and 7s) is itself a lucky number (4 or 7).
 *
 * Algorithm: Counts occurrences of lucky digits in the input number string, then checks if this count is 4 or 7.
 * Time Complexity: O(L) where L is the number of digits in the input number (length of the string).
 * Space Complexity: O(L) for storing the input number as a string.
 */
/// Problem Name: Nearly Lucky Number
/// Problem Link: https://codeforces.com/problemset/problem/110/A

#include <bits/stdc++.h> // Includes most standard C++ libraries, especially <string> and <iostream>.

using namespace std;

/**
 * @brief Main function to check if a number is a Nearly Lucky Number.
 * @pre Input consists of a single number as a string.
 * @post Prints "YES" if the number is nearly lucky, otherwise "NO".
 */
int main() {
  string s; // Variable to store the input number as a string.
  // Functional Utility: Reads the input number string.
  cin >> s;

  int d = 0; // Counter for lucky digits (4s and 7s).

  /**
   * Block Logic: Iterates through each character (digit) of the input string to count lucky digits.
   * Invariant: 'd' increments each time a '4' or '7' is encountered.
   */
  for (int i = 0; i < s.size(); i++) {
    /**
     * Block Logic: Checks if the current digit is a lucky digit.
     * Pre-condition: `s[i]` is the current character being examined.
     * Invariant: If `s[i]` is '4' or '7', it's a lucky digit.
     */
    if (s[i] == '4' || s[i] == '7')
      d++; // Increment lucky digit counter.
  }

  /**
   * Block Logic: Determines if the count of lucky digits ('d') is itself a lucky number.
   * Pre-condition: 'd' holds the total count of lucky digits found in the input string.
   * Invariant: If 'd' is 4 or 7, the original number is "nearly lucky".
   */
  if (d == 4 || d == 7)
    cout << "YES" << '\n'; // Prints "YES" if 'd' is 4 or 7.
  else
    cout << "NO" << '\n'; // Prints "NO" otherwise.

  // Indicates successful program execution.
  return 0;
}
