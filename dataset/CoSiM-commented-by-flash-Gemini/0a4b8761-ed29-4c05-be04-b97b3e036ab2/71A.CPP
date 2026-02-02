/**
 * @file 71A.CPP
 * @brief Solution for Codeforces Problem 71A - Way Too Long Words.
 *
 * This problem asks to abbreviate "long" words (words with length greater than 10).
 * An abbreviation consists of the first letter, the count of letters between the first and last (as a number), and the last letter.
 * Words that are not long enough are printed as they are.
 *
 * Algorithm: String length check and conditional formatting.
 * For each input word, the code checks its length. If the length is greater than 10,
 * it constructs and prints the abbreviated form; otherwise, it prints the word as is.
 *
 * Time Complexity: O(L) per word, where L is the length of the word, due to string operations.
 * If there are N words, total time is O(N * L_avg).
 * Space Complexity: O(L_max) for storing the longest word.
 */

#include <bits/stdc++.h> // Includes most standard C++ libraries, common in competitive programming.
using namespace std; // Functional Utility: Brings all identifiers from the std namespace into the current scope, simplifying code by avoiding explicit std:: prefixes.

/**
 * @brief Main function: Solves the Way Too Long Words problem.
 * Functional Utility: Reads an integer `n` (number of words), then processes `n` words.
 * For each word, it checks its length and prints either the full word or its abbreviation.
 * Algorithm: Conditional string manipulation based on length.
 */
int main() {

  int n; // Declares an integer `n` to store the number of words.
  cin >> n; // Reads the number of words.
  // Block Logic: Loop `n` times to process each word.
  // Precondition: `n` is a non-negative integer.
  // Invariant: Each iteration processes one word.
  while (n--) {
    string s; // Declares a string `s` to store the current word.
    cin >> s; // Reads the current word.
    // Block Logic: Check word length and print either full word or abbreviation.
    if (s.length() > 10) // If the word is "long" (length > 10).
      // Functional Utility: Prints the first character, the length minus 2, and the last character.
      cout << s[0] << (s.length() - 2) << s[s.length() - 1] << endl;
    else // If the word is not "long".
      cout << s << endl; // Prints the word as is.
  }
  return 0; // Indicate successful program execution.
}