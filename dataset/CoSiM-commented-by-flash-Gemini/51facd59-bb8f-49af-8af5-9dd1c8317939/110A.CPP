/**
 * @file 110A.CPP
 * @brief Solution to Codeforces Problem 110A: "Nearly Lucky Number".
 *        This program determines if a given number (represented as a string) is "nearly lucky".
 *        A number is nearly lucky if the count of "lucky" digits (4 or 7) within it is itself a lucky number (4 or 7).
 *
 * Algorithm: String iteration and counting of specific digits. The count is then checked against lucky numbers.
 * Time Complexity: O(L) where L is the length of the input string, due to single pass
 *                  iteration.
 * Space Complexity: O(L) to store the input string.
 */

#include <bits/stdc++.h>

using namespace std;

template <typename T> string NumberToString(T Number) {
  /**
   * Functional Utility: Converts any numeric type to its string representation.
   *                     Although not used in this specific problem's solution, it's
   *                     a common utility function for string-number conversions.
   */
  ostringstream ss;
  ss << Number;
  return ss.str();
}

int main() {
  /**
   * Functional Utility: Orchestrates the process of reading the input number as a string,
   *                     counting its lucky digits, and determining if the count itself is lucky.
   */
  string str;    // Declares a string to store the input number.
  cin >> str;     // Reads the input number as a string from standard input.
  int count = 0;  // Initializes a counter for lucky digits (4s and 7s).
  /**
   * Block Logic: Iterates through each character of the input string to count lucky digits.
   * Pre-condition: `str` contains the input number as a string. `count` is initialized to 0.
   * Invariant: `count` accurately reflects the number of lucky digits encountered so far.
   */
  for (int i = 0; i < str.length(); i++) {
    // Check if the current digit is a lucky digit (4 or 7).
    if (str[i] == '4' || str[i] == '7') {
      count++; // Increment count if the digit is lucky.
    }
  }
  /**
   * Block Logic: Determines if the count of lucky digits is itself a lucky number.
   * Pre-condition: `count` holds the total number of lucky digits in the input string.
   * Invariant: Prints "YES" if `count` is 4 or 7, otherwise prints "NO".
   */
  if (count == 4 || count == 7) {
    cout << "YES" << endl; // Output "YES" if the count of lucky digits is 4 or 7.
  } else {
    cout << "NO" << endl; // Output "NO" otherwise.
  }
  return 0; // Indicate successful program termination.
}
