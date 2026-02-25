/**
 * @file raw/46ae8003-f180-4fa0-9a36-c116cc1f2057/71A.CPP
 * @brief Solution to Codeforces problem 71A - Way Too Long Words.
 *
 * @algorithm
 * The problem requires abbreviating long words. A word is "too long"
 * if its length is strictly greater than 10. The abbreviation rule is to write
 * the first letter, the number of middle letters, and the last letter.
 * For example, "localization" becomes "l10n".
 *
 * The algorithm is:
 * 1. Read `n`, the number of words.
 * 2. Loop `n` times.
 * 3. Read a word `s`.
 * 4. If `s.size() > 10`, print the abbreviation.
 * 5. Otherwise, print the word as is.
 *
 * @note
 * The original code has a line:
 * `cout << s[0] + to_string(s.size() - 2) + s.back() << endl;`
 * This line is syntactically incorrect in standard C++ because `operator+` is not
 * defined for `char + std::string`. It seems to be a typo and was likely intended
 * to be `cout << s[0] << (s.size() - 2) << s.back() << endl;`
 * The documentation explains the intended logic.
 *
 * @complexity
 * - Time Complexity: O(n * L), where `n` is the number of words and L is the
 *   max word length, dominated by input reading.
 * - Space Complexity: O(L), to store the longest word.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  int n; // Number of words.
  cin >> n;
  string s;
  
  // Process each word.
  while (n--) {
    cin >> s;
    
    /**
     * @block
     * @description This conditional block implements the abbreviation logic.
     * It checks if the word is "too long" and applies the abbreviation rule accordingly.
     */
    if (s.size() > 10)
      // This line contains a syntax error in standard C++. The `+` operators
      // are likely intended to be `<<` stream insertion operators.
      // It attempts to print the first letter, the count of middle letters, and the last letter.
      cout << s[0] + to_string(s.size() - 2) + s.back() << endl;
    else
      // Print short words as they are.
      cout << s << endl;
  }
  return 0;
}
