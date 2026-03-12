/**
 * @5c1cff46-bb2b-4dd6-8fe1-b0c6e0073e37/110A.CPP
 * @brief This program solves problem 110A, known as "Nearly Lucky Number".
 * It determines if a given number (read as a string) is "nearly lucky". A number
 * is nearly lucky if the count of its "lucky digits" (4s and 7s) is itself a lucky number
 * (i.e., the count is either 4 or 7).
 *
 * Algorithm: Single pass over the string representation of the number to count lucky digits,
 * followed by a conditional check on the count.
 * Time Complexity: O(L) where L is the length of the input number as a string.
 * Space Complexity: O(L) for storing the input string.
 */

#include <bits/stdc++.h> // A common header in competitive programming; includes iostream, string.
using namespace std;     // Uses the standard namespace.

int main(void) {
  string s; // Declares a string variable to store the input number.
  cin >> s; // Reads the number as a string from standard input.
  int ans = 0; // Initializes 'ans' to 0, which will count the lucky digits.

  /**
   * Block Logic: Iterates through each character (digit) of the input string to count lucky digits.
   * Invariant: 'ans' accumulates the count of '4's and '7's found so far.
   */
  for (int i = 0; i < s.length(); ++i) { // Loop through each character of the string.
    // Inline: Checks if the current character is '7' or '4'.
    if (s[i] == '7' || s[i] == '4') { 
      ans++; // Increments 'ans' if the digit is lucky.
    }
  }
  
  /**
   * Block Logic: Checks if the count of lucky digits ('ans') is itself a lucky number.
   * Pre-condition: 'ans' holds the total count of lucky digits in the input string.
   * Invariant: Prints "YES" if 'ans' is 4 or 7, otherwise "NO".
   */
  // Inline: Uses a ternary operator to conditionally print "YES" if 'ans' is 4 or 7, else "NO".
  // `'\n'` is used for a newline character.
  cout << ((ans == 4 || ans == 7) ? "YES" : "NO") << '\n'; 
  return 0; // Indicates successful program termination.
}
