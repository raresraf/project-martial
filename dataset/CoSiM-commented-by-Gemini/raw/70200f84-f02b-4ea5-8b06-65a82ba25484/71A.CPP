/**
 * @file 71A.CPP
 * @brief Solution for Codeforces Problem 71A - "Way Too Long Words".
 *
 * This program implements a common string manipulation task found in
 * competitive programming. It reads a series of words and, for any word
 * that is longer than 10 characters, it prints an abbreviation. The
 * abbreviation consists of the first letter, the number of letters
 * in between, and the last letter.
 *
 * @algorithm String length check and conditional abbreviation.
 * @time_complexity O(N*L) in the worst case, where N is the number of words
 *                  and L is the length of the longest word, due to reading
 *                  the strings. In practice, the processing for each
 *                  string is O(1) after it is read.
 * @space_complexity O(L) to store the longest word.
 */
// the-Quert 71A - Way Too Long Words
#include <iostream>

using namespace std;

/**
 * @brief Main function to process the words.
 */
int main() {
  int n;
  string s;
  // Read the number of words to process.
  cin >> n;
  /**
   * @brief Loop to process each word.
   * This loop iterates 'n' times, reading a word in each iteration and
   * applying the abbreviation logic.
   */
  while (n--) {
    cin >> s;
    /**
     * @brief Abbreviation logic.
     * If the word's length is greater than 10, it is abbreviated.
     * Otherwise, it is printed as is. The abbreviation is formed by
     * the first character, the count of intermediate characters
     * (length - 2), and the last character.
     */
    if (s.length() > 10) {
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    } else {
      cout << s << endl;
    }
  }
  return 0;
}