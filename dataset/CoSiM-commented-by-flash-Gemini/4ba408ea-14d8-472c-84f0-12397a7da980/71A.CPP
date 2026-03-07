/**
 * @file 71A.CPP
 * @brief Solution to a competitive programming problem ("Way Too Long Words" - Codeforces 71A).
 * The program reads a number of test cases, and for each test case, it processes a single word.
 * If a word's length is greater than 10 characters, it is abbreviated by taking its first character,
 * the count of intermediate characters, and its last character (e.g., "localization" becomes "l10n").
 * Otherwise, the word is printed as is.
 * Algorithm: String length-based conditional abbreviation.
 * Time Complexity: O(T * L), where T is the number of test cases and L is the maximum length of a word.
 * Space Complexity: O(L) for storing the input string per test case.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.
using namespace std;

int main() {
  // Declares 't' to store the number of test cases.
  int t;
  // Reads the number of test cases.
  cin >> t;
  
  // Declares a string 'temp' to store each input word.
  string temp;
  
  /**
   * Block Logic: Iterates 't' times, once for each test case.
   * Invariant: At the start of each iteration, 'i' is the current test case index (0-based).
   * Pre-condition: 't' is a positive integer indicating the number of words to follow.
   */
  for (int i = 0; i < t; ++i) {
    // Reads the word for the current test case.
    cin >> temp;
    // Calculates the length of the current word.
    int l = temp.length();
    
    /**
     * Conditional Logic: Checks if the length of the current word is within the limit (10 characters or less).
     * Post-condition: Either the original word or its abbreviated form is printed.
     */
    if (l <= 10) {
      // If the word is short enough, prints it as is.
      cout << temp << endl;
    } else {
      // If the word is too long, prints its first character, the count of intermediate characters, and its last character.
      // (l - 2) represents the count of characters between the first and last.
      cout << temp[0] << l - 2 << temp[l - 1] << endl;
    }
  }
  
  return 0; // Indicates successful program execution.
}