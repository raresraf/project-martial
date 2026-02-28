/**
 * @file 71A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 71A - "Way Too Long Words").
 *
 * This program implements a word abbreviation rule. If a word's length exceeds 10 characters,
 * it is abbreviated to its first character, followed by the count of characters between
 * the first and last, and then its last character (e.g., "localization" becomes "l10n").
 * Words with 10 or fewer characters are printed as they are.
 *
 * Algorithm: String length check and conditional abbreviation.
 * For each input word, its length is calculated. Based on whether the length is
 * greater than 10, either the original word or its abbreviated form is printed.
 *
 * Time Complexity: O(L) for each test case, where L is the length of the word.
 * This includes reading the word, calculating its length (strlen), and printing.
 * Total time complexity is O(T * L_max), where T is the number of test cases and L_max is the maximum word length.
 * Space Complexity: O(L_max) for storing the input word.
 */
#include <bits/stdc++.h> // Includes various standard libraries for C++, including cstring for strlen.

using namespace std;

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads the number of test cases, then processes each word according to the abbreviation rules.
 */
int main() {
  int t;         // Variable to store the number of test cases.
  int len = 0;   // Variable to store the length of the current word.
  char word[101]; // Character array to store the input word (max 100 chars + null terminator).

  // Pre-condition: Read the number of test cases.
  cin >> t;
  // Block Logic: Process each test case.
  // Invariant: 't' decreases with each iteration, loop continues until all test cases are processed.
  while (t--) {
    // Pre-condition: Read the current word.
    cin >> word;
    // Functional Utility: Calculate the length of the word.
    len = strlen(word);
    // Block Logic: Apply the abbreviation rule based on word length.
    if (len <= 10) {
      // If the word is short enough, print it as is.
      cout << word << '\n';
    } else {
      // If the word is too long, print its abbreviated form.
      // First character, then (length - 2), then last character.
      cout << word[0] << len - 2 << word[len - 1] << '\n';
    }
  }
  return 0; // Indicate successful execution.
}
