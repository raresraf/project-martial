/**
 * @file 71A.CPP
 * @brief This program solves the "Way Too Long Words" problem (Codeforces 71A).
 *   It abbreviates words longer than 10 characters to their first letter,
 *   the count of intermediate characters, and their last letter. Shorter words are printed as is.
 *   Problem description: https://codeforces.com/problemset/problem/71/A
 * * Algorithm: String length check and conditional formatting.
 *   1. Reads all `N` words into an array of strings.
 *   2. In a separate loop, iterates through the stored words.
 *   3. For each word, if its length is greater than 10, it prints the first character,
 *      then its length minus 2, then the last character. Otherwise, it prints the word as is.
 * Time Complexity: O(N * L_max) where N is the number of words and L_max is the maximum length of a word.
 *   The first loop for reading words and the second loop for processing words both contribute linearly
 *   to the total time complexity.
 * Space Complexity: O(N * L_max) to store the `s` array of strings. For `N` up to 100 and `L_max` up to 100,
 *   this is `100 * 100` characters, which is feasible.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, such as <iostream>, <string>, <vector>.
#include <iostream>  // Required for std::cin, std::cout, std::endl.
using namespace std;

/**
 * @brief Main function to process words and apply abbreviation rules.
 * Functional Utility: Reads the number of words, then all words into an array,
 *                     processes each word for abbreviation, and prints the result.
 */
int main() {
  int n; // Stores the number of words.
  // `s`: An array of `std::string` to store the input words. Max capacity 100 words.
  string s[100];
  // Block Logic: Reads the total number of words, `n`.
  cin >> n;
  /**
   * Block Logic: Reads `n` words from input and stores them in the string array `s`.
   * Pre-condition: `n` is a non-negative integer.
   * Invariant: `s` array is populated with all input words.
   */
  for (int i = 0; i < n; i++) {
    cin >> s[i];
  }
  /**
   * Block Logic: Iterates through the stored words, applying the abbreviation logic to each.
   * Pre-condition: `s` array is populated with words.
   * Invariant: Each word `s[i]` is processed and printed according to its length.
   */
  for (int i = 0; i < n; i++) {
    /**
     * Conditional Logic: Checks if the current word's length is greater than 10.
     * If true, it prints the abbreviated form: first character + (length - 2) + last character.
     * Otherwise, it prints the original word.
     */
    if (s[i].length() > 10)
      cout << s[i][0] << s[i].length() - 2 << s[i][s[i].length() - 1] << endl;
    else
      cout << s[i] << endl;
  }
  return 0;
}
