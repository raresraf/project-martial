/**
 * @file 71A.CPP
 * @brief Solution for a string abbreviation problem from a programming contest.
 * @details The program processes a list of words. If a word is longer than 10 characters,
 * it is abbreviated by taking its first letter, the count of letters in between, and its
 * last letter. For example, "localization" becomes "l10n". Shorter words (<=10 characters)
 * are printed unchanged.
 *
 * Algorithm:
 * 1. Read the number of words, 'n'.
 * 2. Loop 'n' times to process each word:
 *    a. Read the word.
 *    b. Check if its length is greater than 10.
 *    c. If it is, construct the abbreviation:
 *       - Get the first character.
 *       - Calculate the number of middle characters (length - 2).
 *       - Get the last character.
 *       - Concatenate and print these three parts.
 *    d. If the length is not greater than 10, print the word as is.
 *
 * Time Complexity: O(N * L_avg), where N is the number of words and L_avg is their average length.
 * Space Complexity: O(L_max), where L_max is the length of the longest word.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  int n;
  cin >> n;
  string s;
  
  /**
   * @block
   * @description This loop iterates 'n' times, processing one word in each iteration.
   */
  while (n--) {
    cin >> s;
    
    /**
     * @block
     * @description This conditional block implements the abbreviation logic.
     * It checks if the word is "too long" and applies the abbreviation rule accordingly.
     */
    if (s.size() > 10)
      // Construct and print the abbreviation for long words.
      cout << s[0] + to_string(s.size() - 2) + s.back() << endl;
    else
      // Print short words as they are.
      cout << s << endl;
  }
  return 0;
}
