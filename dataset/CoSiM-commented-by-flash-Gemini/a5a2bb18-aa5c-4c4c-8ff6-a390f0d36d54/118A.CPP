/**
 * @file 118A.CPP
 * @brief Logic for string transformation through vocalic deletion and consonant normalization.
 * 
 * Logic: Implements a specific string processing task where all vowels are removed, 
 * and remaining consonants are prefixed with a dot and converted to lowercase.
 * 
 * Algorithm: Single-pass linear scan with conditional filtering.
 * Time Complexity: $O(N)$ where N is the length of the input string.
 * Space Complexity: $O(N)$ to store the transformed output sequence.
 */

#include <iostream>
#include <string>
using namespace std;

int main() {
  string inp, out = "";
  // Functional Utility: Reads the source string for processing.
  cin >> inp;

  /**
   * Block Logic: Iterates through the input string to apply transformation rules.
   * Invariant: Characters are processed in their original sequential order.
   */
  for (int i = 0; i < inp.size(); i++) {
    /**
     * Logic: Vowel Detection (Case-Insensitive).
     * Condition: If the character is any of {a, i, e, o, u, y}, it is skipped.
     */
    if (inp[i] == 'a' || inp[i] == 'A' || inp[i] == 'i' || inp[i] == 'I' ||
        inp[i] == 'e' || inp[i] == 'E' || inp[i] == 'o' || inp[i] == 'O' ||
        inp[i] == 'u' || inp[i] == 'U' || inp[i] == 'y' || inp[i] == 'Y') {
      continue;
    } else {
      /**
       * Functional Utility: Consonant processing.
       * 1. Appends the '.' separator.
       * 2. Appends the character normalized to lowercase using tolower().
       */
      out += ".";
      out += (char)tolower(inp[i]);
    }
  }
  
  // Output the final transformed string.
  cout << out << endl;
}
