/**
 * @file 118A.CPP
 * @brief An alternative solution to Codeforces Problem 118A - String Task.
 * @details This program transforms a string by removing vowels, converting consonants to lowercase,
 * and prepending a '.' to each consonant. This implementation uses a multi-pass approach.
 * Vowels are defined as 'A', 'O', 'Y', 'E', 'U', 'I', and their lowercase counterparts.
 *
 * @author (unknown)
 *
 * @algorithm
 * The solution is implemented in three main passes:
 * 1. **Vowel Removal Pass:** The code iterates through the input string and removes all vowel
 *    characters (both uppercase and lowercase). The string is modified in-place.
 * 2. **Lowercase Conversion Pass:** It then iterates through the resulting string (now only
 *    consonants) and converts any uppercase characters to lowercase.
 * 3. **Formatting Pass:** Finally, it iterates through the all-consonant, lowercase string
 *    and builds a new string, prepending a '.' to each character.
 * The final formatted string is then printed.
 *
 * @complexity
 * Time Complexity: O(N^2) in the worst case for the vowel-removal loop, because `std::string::erase`
 * can take O(N) time. The subsequent loops are O(N).
 * Space Complexity: O(N) to store the input string and the final result string.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {

  string word1, word2;
  cin >> word1;

  /**
   * Block Logic: First pass - Vowel Removal.
   * This loop iterates through the string to identify and remove all vowels.
   * When a vowel is found, `word1.erase(i, 1)` removes it. `i--` is crucial
   * to re-examine the character that shifts into the current index `i`.
   */
  for (int i = 0; i < word1.length(); i++) {
    // "A", "O", "Y", "E", "U", "I"

    if (word1[i] == 'a' || word1[i] == 'A' || word1[i] == 'o' ||
        word1[i] == 'O' || word1[i] == 'y' || word1[i] == 'Y' ||
        word1[i] == 'E' || word1[i] == 'e' || word1[i] == 'u' ||
        word1[i] == 'U' || word1[i] == 'i' || word1[i] == 'I') {
      word1.erase(i, 1);
      i--;
    }
  }

  /**
   * Block Logic: Second pass - Lowercase Conversion.
   * This loop iterates through the modified string, which now only contains consonants,
   * and converts any uppercase letters to their lowercase equivalents.
   */
  for (int i = 0; i < word1.length(); ++i) {
    if (isupper(word1[i]))
      word1[i] = tolower(word1[i]);
  }

  /**
   * Block Logic: Third pass - Formatting.
   * This loop constructs the final output string `word2` by prepending a '.'
   * to each character of the consonant-only, all-lowercase string `word1`.
   */
  for (int j = 0; j < word1.length(); j++) {
    word2 = word2 + "." + word1[j];
  }
  cout << word2 << endl;

  return 0;
}