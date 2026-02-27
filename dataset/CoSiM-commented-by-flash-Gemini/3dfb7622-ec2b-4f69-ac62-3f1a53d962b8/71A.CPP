/**
 * @file 71A.CPP
 * @brief Solution for competitive programming problem: "Way Too Long Words".
 *
 * @details This program processes a list of words. For each word, if its length
 *          is greater than 10 characters, it is replaced by an abbreviation.
 *          The abbreviation consists of the first character, followed by the
 *          count of characters between the first and last characters, and then
 *          the last character. Words with length 10 or less are unchanged.
 *          Example: "localization" becomes "l10n".
 *          Algorithm: Conditional string length check and abbreviation.
 *          Time Complexity: O(T * L), where T is the number of test cases (words)
 *                           and L is the maximum length of a word. String operations
 *                           are typically O(L).
 *          Domain: Competitive Programming.
 */
#include <iostream>
#include <string>
using namespace std;
int main() {
  // Declare 't' for the number of test cases (words to process).
  int t;
  // Read the number of test cases.
  cin >> t;
  
  // Block Logic: Iterate 't' times, processing one word per iteration.
  // Invariant: Each iteration reads a word, checks its length, and prints either the original or abbreviated form.
  for (int i = 0; i < t; i++) {
    string s; // Declare string 's' to store the current word.
    cin >> s;  // Read the word.
    
    // Precondition: If the word's length is greater than 10, apply abbreviation.
    if (s.length() > 10)
      // Print first character, then the count of inner characters (length - 2), then the last character.
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    else
      // If the word is 10 characters or less, print it as is.
      cout << s << endl;
  }
}
