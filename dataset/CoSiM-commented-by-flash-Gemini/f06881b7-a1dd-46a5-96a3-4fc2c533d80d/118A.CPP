/**
 * @file 118A.CPP
 * @brief Processes an input string by converting it to lowercase, removing vowels,
 *        and then inserting a '.' before each remaining consonant.
 * Algorithm: String manipulation involving case conversion, regular expression-based
 *            vowel removal, and iterative string construction.
 * Time Complexity: O(L) where L is the length of the input string, dominated by
 *                  the iteration for case conversion and string building, and
 *                  regex replacement complexity which is typically proportional to string length.
 * Space Complexity: O(L) for storing the modified strings.
 */
//
// Created by Taewoo Kim on 2/7/2017.
//

#include <iostream> // For standard input/output operations (cin, cout).
#include <string>   // For string manipulation.
#include <regex>    // For regular expression operations (regex_replace).
#include <cctype>   // For tolower function.

using namespace std;

/**
 * @brief Main function to perform string processing as described.
 * Functional Utility: Takes a word, normalizes it, filters out specific characters,
 *                     and reformats it according to a pattern.
 */
int main() {
  string s;
  // Block Logic: Reads a single string (word) from standard input.
  cin >> s;
  
  // Block Logic: Converts all characters in the input string to lowercase.
  // Invariant: After this loop, 's' will contain only lowercase characters.
  for (int i = 0; i < s.size(); i++) {
    s[i] = tolower(s[i]);
  }
  
  // Block Logic: Removes all occurrences of vowels (a, e, i, o, u, y) from the string.
  // Functional Utility: Filters out specific characters using a regular expression.
  // Note: 'y' is treated as a vowel in this problem context.
  s = regex_replace(s, regex("[aeiouy]"), "");
  
  string res = ""; // Initializes an empty string to build the result.
  // Block Logic: Iterates through the modified string (consonants only)
  //              and prepends a '.' before each character to form the final result.
  // Invariant: 'res' accumulates characters from 's' with a '.' prefix.
  for (char c : s) {
    res += ".";
    res += c;
  }
  
  // Functional Utility: Prints the final processed string to standard output.
  cout << res;
  return 0; // Indicates successful program termination.
}