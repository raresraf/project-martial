#include <iostream>
#include <string>
#include <cctype> // For std::tolower
using namespace std;

/**
 * @file 118A.CPP
 * @brief Solution for the "String Task" problem.
 *
 * Problem Statement:
 * Given a string, apply the following transformations:
 * 1. Delete all vowels. The vowels are defined as 'A', 'O', 'Y', 'E', 'U', 'I',
 *    and their lowercase counterparts.
 * 2. Insert a '.' character before each consonant.
 * 3. Replace all uppercase consonants with their lowercase versions.
 *
 * Algorithm:
 * This solution uses a two-pass approach.
 * 1.  **First Pass (Normalization)**: The entire input string is converted to
 *     lowercase in-place.
 * 2.  **Second Pass (Transformation)**: It iterates through the lowercase
 *     string. For each character, it checks if it's a vowel.
 *     - If it is a vowel, it is ignored (effectively deleted).
 *     - If it is a consonant, a '.' followed by the character is appended to
 *       a result string.
 * 3. Finally, the result string is printed.
 *
 * Time Complexity: O(L), where L is the length of the string, due to the two
 * passes.
 * Space Complexity: O(L) to store the input and result strings.
 */

int main() {
  // A common competitive programming optimization to speed up C++ I/O.
  ios::sync_with_stdio(false);

  string s, res;
  cin >> s;

  // First Pass: Convert the entire string to lowercase.
  for (int i = 0; i < (int)s.size(); ++i) {
    // The expression `s[i] - ('A' - 'a')` is equivalent to `s[i] + 32`,
    // which correctly converts from uppercase to lowercase in ASCII.
    // A more readable alternative is `s[i] = tolower(s[i]);`
    if (s[i] >= 'A' && s[i] <= 'Z') {
      s[i] = s[i] - ('A' - 'a');
    }
  }

  // Second Pass: Build the result string.
  // Invariant: 'res' contains the transformed version of the first 'i' characters of 's'.
  for (int i = 0; i < (int)s.size(); ++i) {
    // Pre-condition: Check if the character is a consonant.
    if (s[i] != 'a' && s[i] != 'o' && s[i] != 'y' && s[i] != 'e' &&
        s[i] != 'u' && s[i] != 'i') {
      // If it's a consonant, append a dot and the character to the result.
      // `string() + "."` is used to ensure string concatenation.
      // A simpler way would be `res += '.'; res += s[i];`
      res += '.';
      res += s[i];
    }
  }

  cout << res << endl;
  return 0;
}
