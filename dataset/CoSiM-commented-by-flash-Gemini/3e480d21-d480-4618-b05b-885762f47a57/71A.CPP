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
 *          Time Complexity: O(N * L), where N is the number of test cases (words)
 *                           and L is the maximum length of a word. String operations
 *                           are typically O(L).
 *          Domain: Competitive Programming.
 */
// http://codeforces.com/problemset/problem/71/A

#include <iostream>
using namespace std;

int main() {
  // Declare 'n' for the number of test cases (words to process).
  // 'le' for storing the length of the current word.
  int n, le;
  // Declare string 's' to store the current word.
  string s;
  // Read the number of test cases.
  cin >> n;
  
  // Block Logic: Iterate 'n' times, processing one word per iteration.
  // Invariant: Each iteration reads a word, checks its length, and prints either the original or abbreviated form.
  while (n--) {
    cin >> s;        // Read the word.
    le = s.length(); // Get the length of the current word.
    
    // Precondition: If the word's length is greater than 10, apply abbreviation.
    if (le > 10)
      // Print first character, then the count of inner characters (length - 2), then the last character.
      cout << s[0] << le - 2 << s[le - 1] << endl;
    else
      // If the word is 10 characters or less, print it as is.
      cout << s << endl;
  }
  return 0; // Indicate successful program termination.
}
