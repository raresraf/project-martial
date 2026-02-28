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
 * For each input word, its length is calculated using `std::string::length()`.
 * Based on whether the length is greater than 10, either the original word or
 * its abbreviated form is constructed and printed.
 *
 * Time Complexity: O(L) for each test case, where L is the length of the word.
 * Reading the word takes O(L), and accessing characters/printing is O(1) or O(L).
 * Total time complexity is O(T * L_max), where T is the number of test cases and L_max is the maximum word length.
 * Space Complexity: O(L_max) for storing the input word in a `std::string`.
 */
#include <bits/stdc++.h> // Includes various standard libraries for C++.
using namespace std;

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads the number of test cases, then processes each word according to the abbreviation rules.
 */
int main() {
  int t; // Variable to store the number of test cases.
  // Pre-condition: Read the number of test cases.
  cin >> t;
  // Block Logic: Process each test case.
  // Invariant: 't' decreases with each iteration, loop continues until all test cases are processed.
  while (t--) {
    string s; // Variable to store the input word.
    // Pre-condition: Read the current word.
    cin >> s;
    int n = s.length(); // Functional Utility: Get the length of the string.
    // Block Logic: Apply the abbreviation rule based on word length.
    if (n <= 10) {
      // If the word is short enough, print it as is.
      cout << s << endl; // Use endl for flushing output.
    } else {
      // If the word is too long, print its abbreviated form.
      // First character, then (length - 2), then last character.
      cout << s[0] << n - 2 << s[n - 1] << endl; // Use endl for flushing output.
    }
  }
  return 0; // Indicate successful execution.
}
