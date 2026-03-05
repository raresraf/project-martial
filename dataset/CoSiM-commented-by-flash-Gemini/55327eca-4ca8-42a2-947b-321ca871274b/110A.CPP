/**
 * @file 110A.CPP
 * @brief Solution to the "Nearly Lucky Number" problem. A number is considered
 *        "nearly lucky" if the count of its "lucky digits" (4s and 7s) is itself
 *        a lucky number (a number containing only 4s and 7s).
 *
 * Algorithm: Digit counting and regex pattern matching.
 *            First, the program counts the occurrences of digits '4' and '7'
 *            in the input number. Then, it converts this count into a string
 *            and uses a regular expression to check if this count-string
 *            consists solely of digits '4' and '7'.
 * Time Complexity: O(log10(N)), where N is the input number.
 *                  The complexity is dominated by converting N to a string and
 *                  iterating through its digits, and then converting the count
 *                  back to a string and matching a regex. The number of digits
 *                  is proportional to log10(N).
 * Space Complexity: O(log10(N)) for string conversions.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  long long num;
  // Block Logic: Reads the input number.
  cin >> num;

  // Functional Utility: Converts the input number to its string representation
  // to facilitate digit-by-digit processing.
  string str = to_string(num);
  int count = 0; // Functional Utility: `count` stores the number of lucky digits (4s and 7s).
  /**
   * Block Logic: Iterates through each character (digit) of the number's string representation.
   * Invariant: `c` is the current digit character, `count` accumulates the lucky digit occurrences.
   */
  for (char c : str) {
    // Functional Utility: Converts the character digit to its integer value.
    // ASCII '0' is 48, so subtracting 48 gives the integer value.
    int l = (int)c - 48;

    // Pre-condition: Checks if the current digit is either 4 or 7.
    if (l == 4 || l == 7)
      // Block Logic: Increments the lucky digit counter if the condition is met.
      count++;
  }

  // Functional Utility: Defines a regular expression pattern to match strings
  // that consist only of '4's and '7's.
  regex pattern("[47]*");
  // Functional Utility: Converts the `count` of lucky digits into a string.
  string nl = to_string(count);

  // Functional Utility: Uses a ternary operator to perform regex matching and
  // print "YES" or "NO" based on the result.
  // Pre-condition: Checks if the string representation of `count` matches the lucky digits pattern.
  regex_match(nl, pattern) ? cout << "YES" : cout << "NO";
  // Block Logic: Prints a newline character after the result.
  cout << endl;

  return 0;
}
