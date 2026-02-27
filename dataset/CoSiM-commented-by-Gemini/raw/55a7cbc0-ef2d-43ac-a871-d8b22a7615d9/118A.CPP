/**
 * @file 118A.CPP
 * @brief An incorrect attempt to solve the "String Task" problem from Codeforces.
 * @details The problem requires deleting vowels, prefixing consonants with '.',
 * and converting them to lowercase. This implementation contains a major logical
 * flaw that prevents it from correctly identifying and deleting vowels.
 *
 * @algorithm (as implemented)
 * 1. Read the input string.
 * 2. Loop through each character of the string.
 * 3. **Logical Flaw**: The code checks if the current character is a vowel.
 *    However, the `if` block for vowels is empty and does not use `continue`
 *    or `break`. The code execution "falls through" to the `else` block
 *    regardless of whether the character is a vowel or a consonant.
 * 4. As a result, every character is treated as a consonant:
 *    a. It is prefixed with a '.'.
 *    b. It is converted to lowercase (if it isn't already).
 *    c. It is printed to the console.
 *
 * @correct_algorithm
 * A correct implementation would ensure that the `else` block is only executed
 * for consonants, for example by putting a `continue;` statement inside the
 * `if` block for vowels.
 *
 * @complexity
 * Time Complexity: O(L), for a single pass over the string of length L.
 * Space Complexity: O(L) to store the input string.
 */
#include <iostream>
using namespace std;

int main() {
  string s;
  cin >> s;
  int len = s.length();
  
  /**
   * @block
   * @brief Process each character of the string.
   */
  for (int i = 0; i < len; i++) {
    /**
     * @warning This `if` block contains a critical bug. When a vowel is found,
     * it enters an empty scope `{ int i = 0; }` and then execution continues
     * to the `else` block below. It fails to skip the vowels.
     */
    if (s[i] == 'a' || s[i] == 'y' || s[i] == 'Y' || s[i] == 'e' ||
        s[i] == 'i' || s[i] == 'o' || s[i] == 'u' || s[i] == 'A' ||
        s[i] == 'E' || s[i] == 'I' || s[i] == 'O' || s[i] == 'U') {
      // This block is effectively empty and does not prevent the `else` from running.
      int i = 0; 
    }
    // Due to the bug above, this `else` block is executed for ALL characters.
    else {
      // Check if the character is uppercase using ASCII values.
      if (s[i] >= 65 && s[i] <= 90) {
        // Convert to lowercase by adding 32 to the ASCII value.
        s[i] = s[i] + 32;
        cout << "." << s[i];
      }
      else // If already lowercase, just print.
        cout << "." << s[i];
    }
  }

  return 0;
}
