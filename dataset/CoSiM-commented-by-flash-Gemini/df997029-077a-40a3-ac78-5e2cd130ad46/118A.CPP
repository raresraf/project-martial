/**
 * @file 118A.CPP
 * @brief This program transforms an input string by converting all uppercase letters to lowercase,
 * removing specific vowels ('a', 'o', 'y', 'e', 'u', 'i'), and prefixing each remaining consonant with a dot.
 * Algorithm: Iterative string processing with character-by-character transformation.
 * Time Complexity: O(N), where N is the length of the input string, due to two passes.
 * Space Complexity: O(N) for storing the transformed string.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  // Optimizes C++ standard streams for faster input/output operations.
  ios::sync_with_stdio(false);
  string s, res; // 's' stores the input string, 'res' stores the transformed result.
  // Reads the input string.
  cin >> s;
  /**
   * Block Logic: Converts all uppercase letters in the input string 's' to lowercase.
   * Invariant: After this loop, 's' will only contain lowercase letters (and any non-alphabetic characters).
   */
  for (int i = 0; i < (int)s.size(); ++i)
    if (s[i] >= 'A' && s[i] <= 'Z')
      s[i] = s[i] - ('A' - 'a');
  /**
   * Block Logic: Iterates through the lowercase string, filters out specific vowels,
   * and prefixes remaining consonants with a dot, appending them to 'res'.
   * Invariant: 'res' contains only consonants from the original string, each prefixed by '.'.
   */
  for (int i = 0; i < (int)s.size(); ++i) {
    if (s[i] != 'a' && s[i] != 'o' && s[i] != 'y' && s[i] != 'e' &&
        s[i] != 'u' && s[i] != 'i') {
      res += string() + "." + s[i];
    }
  }

  // Prints the final transformed string.
  cout << res << endl;
  // Standard program exit.
  return 0;
}