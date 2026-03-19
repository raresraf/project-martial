/**
 * @file raw/c5fcebce-e0b9-4a53-bad4-2e5a180c1f1a/71A.CPP
 * @brief Implements the "Way Too Long Words" abbreviation logic.
 *
 * This program reads a list of words. For each word, it checks if its length
 * is greater than 10. If it is, the word is replaced with a special abbreviation:
 * the first letter, followed by the number of letters in between, and then the
 * last letter. For example, "localization" becomes "l10n". Otherwise, the word
 * is printed as is.
 *
 * @algorithm
 * 1. Read the total number of words (test cases), `t`.
 * 2. Loop `t` times to process each word.
 * 3. In each iteration, read a word.
 * 4. Calculate the length of the word.
 * 5. If the length is greater than 10, print the first character, the length
 *    minus two, and the last character.
 * 6. If the length is 10 or less, print the word unchanged.
 *
 * @time_complexity O(T * L), where T is the number of words and L is the
 *                  average length of a word, due to `strlen` being called
 *                  multiple times within the loop.
 * @space_complexity O(L_max), where L_max is the maximum possible word length,
 *                   to store the word in a character array.
 */
#include <cstring>
#include <iostream>

using namespace std;
int main() {
  int t;
  char str[1000];
  cin >> t;

  /**
   * @block
   * Main loop to process each of the `t` test cases (words).
   */
  for (; t > 0; t--) {
    cin >> str;
    /**
     * @block
     * Core logic for abbreviation. Checks if the word's length exceeds 10.
     */
    if (strlen(str) <= 10) {
      // If the word is not "too long", print it as is.
      cout << str << endl;
    } else {
      // If the word is "too long", print its abbreviation.
      // Format: [first letter][number of letters between][last letter]
      cout << str[0] << strlen(str) - 2 << str[strlen(str) - 1] << endl;
    }
  }
  return 0;
}
