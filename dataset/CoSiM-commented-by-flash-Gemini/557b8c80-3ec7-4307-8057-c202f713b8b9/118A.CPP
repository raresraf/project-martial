/**
 * @file 118A.CPP
 * @brief Solution to Codeforces Problem 118A: "String Task".
 *        The problem involves transforming a given input string according to specific rules:
 *        1. Delete all vowels (A, O, Y, E, U, I), case-insensitive.
 *        2. Insert a "." before each consonant.
 *        3. Convert all consonants to lowercase.
 *
 * Algorithm: String iteration, transformation, and filtering. The program first converts the entire
 *            input string to lowercase. Then, it iterates through each character. If a character
 *            is identified as a vowel, it is skipped. Otherwise (if it's a consonant), a '.' is
 *            appended to a `stringstream`, followed by the consonant. Finally, the content of
 *            the `stringstream` is extracted into a string and printed.
 * Time Complexity: O(L) where L is the length of the input string, due to one pass for
 *                  case conversion and another pass for filtering and building the result string.
 * Space Complexity: O(L) to store the input string and the `stringstream` output.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * Functional Utility: Declaration of variables for string processing.
   * `a`: Stores the input string.
   * `ss`: A `stringstream` used to efficiently build the output string.
   * `ac`: Temporary character variable to hold the current character being processed.
   * `v`: A C-style array storing lowercase vowels for efficient lookup.
   */
  string a;
  stringstream ss;
  cin >> a; // Input: Read the input string.
  char ac;
  char v[6] = {'a', 'o', 'y', 'e', 'u', 'i'}; // Define array of lowercase vowels.

  /**
   * Block Logic: Converts all characters in the input string `a` to lowercase.
   * Algorithm: `std::for_each` applies a lambda function to each character.
   * Functional Utility: Ensures case-insensitive processing of vowels and consonants.
   */
  for_each(begin(a), end(a), [](char &c) { c = tolower(c); });

  /**
   * Block Logic: Iterates through the lowercase input string `a` to filter characters.
   * Pre-condition: `a` is the lowercase input string. `ss` is empty. `v` contains vowels.
   * Invariant: `ac` holds the current character; `ss` accumulates transformed characters.
   */
  for (int i = 0; a[i] != '\0'; i++) {
    ac = a[i]; // Get the current character.
    /**
     * Conditional Logic: Checks if the current character `ac` is a vowel.
     * Algorithm: `std::find` searches for `ac` within the `v` (vowels) array.
     * `ia` is true if `ac` is a vowel, false otherwise.
     */
    bool ia = find(begin(v), end(v), ac) != end(v);
    /**
     * Conditional Logic: If the character is not a vowel (i.e., it's a consonant).
     * Pre-condition: `ac` is a consonant.
     * Post-condition: A '.' and the consonant `ac` are appended to `ss`.
     */
    if (!ia)
      ss << '.' << ac; // Append '.' and the consonant to the stringstream.
  }
  ss >> a;   // Extract the built string from the stringstream back into `a`.
  cout << a << endl; // Output: Print the final transformed string.
  return 0;
}
