/**
 * @file 71A.CPP
 * @brief This program implements the "Way Too Long Words" problem.
 * It abbreviates words longer than 10 characters by replacing the middle characters
 * with their count, keeping the first and last characters. Words 10 characters or shorter are unchanged.
 * Algorithm: String length check and conditional printing/abbreviation using custom integer-to-string conversion.
 * Time Complexity: O(N * L) where N is the number of words and L is the average length of a word.
 * Space Complexity: O(L) for storing the input string and intermediate string conversions.
 */
#include <algorithm> // Not directly used in this code, but included in original.
#include <cmath>     // Not directly used in this code, but included in original.
#include <cstdio>    // Not directly used in this code, but included in original (e.g., for printf).
#include <cstring>   // Not directly used in this code, but included in original (e.g., for strlen).
#include <iostream>  // Required for cin, cout.
using namespace std;

int n; // Stores the number of words to process.
string s; // Stores the current word being processed.

/**
 * @brief Converts an integer to its string representation.
 * This custom implementation builds the string by repeatedly taking the modulo 10
 * and dividing by 10, then prepending the digit to the result string.
 * @param cur The integer to convert.
 * @return The string representation of the integer.
 */
string convert(int cur) {
  string res;
  while (cur > 0) {
    // Extracts the last digit and converts it to a character, then prepends it to 'res'.
    res = (char)('0' + cur % 10) + res;
    // Removes the last digit from 'cur'.
    cur /= 10;
  }
  return res;
}

int main() {
  // Reads the number of words that will follow.
  cin >> n;
  /**
   * Block Logic: Processes 'n' words, one in each iteration, applying the abbreviation rule.
   * Invariant: Each iteration handles one word, and 's' holds the current word.
   */
  for (int i = 0; i < n; ++i) {
    // Reads the current word into 's'.
    cin >> s;
    /**
     * Conditional Logic: Checks the length of the current word to decide whether to abbreviate.
     * If the word's length is 10 or less, it's printed as is.
     * If the word's length is greater than 10, it's abbreviated.
     */
    if (s.length() <= 10)
      // Prints the word as it is, followed by a newline.
      cout << s << endl;
    else {
      // Constructs the abbreviated string: first character + (length - 2) as string + last character.
      string t = s[0] + convert(s.length() - 2) + s[s.length() - 1];
      // Prints the abbreviated word, followed by a newline.
      cout << t << endl;
    }
  }

  // Standard program exit.
  return 0;
}
