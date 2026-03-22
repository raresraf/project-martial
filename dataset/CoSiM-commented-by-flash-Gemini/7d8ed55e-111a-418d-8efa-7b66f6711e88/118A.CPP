/**
 * @file 118A.CPP
 * @brief Solution to Codeforces problem 118A - String Task.
 *        This program processes an input string by transforming it according to specific rules:
 *        1. All uppercase Latin letters are converted to their lowercase equivalents.
 *        2. All vowels ('a', 'o', 'y', 'e', 'u', 'i') are removed.
 *        3. A '.' character is inserted before each consonant.
 *
 * Algorithm:
 *   1. Reads an input string `s`.
 *   2. Converts the entire string `s` to lowercase.
 *   3. Iterates through the lowercase string `s`, filtering out vowels.
 *   4. Stores the remaining characters (consonants) in a character vector `m`.
 *   5. Iterates through `m`, printing a '.' followed by each character.
 *
 * Time Complexity: O(L), where L is the length of the input string `s`.
 *                  This includes a linear pass for `transform` and two linear passes for loops.
 * Space Complexity: O(L) for storing the input string `s` and the `vector<char> m`.
 */

#include <bits/stdc++.h> // Includes common C++ standard library headers.
using namespace std;
#define ll long long int // Typedef for long long int.

/**
 * @brief Checks if a given character is a vowel (a, e, i, o, u, y).
 *
 * This function uses a switch statement to efficiently determine if a character
 * is one of the defined vowels. The check is case-sensitive, so the input
 * character should ideally be lowercase.
 *
 * @param c The character to check.
 * @return bool True if the character is a vowel, false otherwise.
 */
bool vowel(char c) {
  switch (c) {
  case 'a':
  case 'e':
  case 'i':
  case 'o':
  case 'u':
  case 'y':
    return true;
  default:
    return false;
  }
}
int main() {
  /**
   * @brief Main function that solves the "String Task" problem.
   *
   * Reads an input string, converts it to lowercase, filters out vowels,
   * and prints the remaining consonants prefixed with a dot.
   *
   * @return int Returns 0 upon successful execution.
   */
  string s;
  cin >> s; // Reads the input string.
  // Transforms all characters in the string `s` to lowercase.
  transform(s.begin(), s.end(), s.begin(), ::tolower);
  
  vector<char> m; // Vector to store the consonants.
  /**
   * Block Logic: Iterates through the lowercase string to identify and collect consonants.
   * Invariant: `m` contains all consonants encountered so far from `s`.
   */
  for (int i = 0; i < s.length(); i++) {
    // Conditional Logic: If the current character is not a vowel, add it to the vector.
    if (!vowel(s[i]))
      m.push_back(s[i]); // Adds the consonant to the vector.
  }
  /**
   * Block Logic: Prints the collected consonants, each prefixed with a dot.
   * Pre-condition: `m` contains the filtered consonants.
   */
  for (int i = 0; i < m.size(); i++) {
    cout << "." << m[i]; // Prints a dot followed by the consonant.
  }
  // Implicitly prints a newline at the end of output due to competitive programming usual practice.
  // Explicit `endl` might be missing but often implied by problem setters.
}
