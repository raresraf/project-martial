/**
 * @file 118A.CPP
 * @brief A partially incorrect solution to the "String Task" problem (Codeforces 118A).
 *
 * @algorithm
 * The problem requires processing a string with the following steps:
 * 1. Convert all uppercase letters to lowercase.
 * 2. Delete all vowels ('a', 'o', 'y', 'e', 'u', 'i').
 * 3. Before each remaining character (a consonant), insert a ".".
 *
 * This code attempts to achieve this by:
 * 1. Removing all uppercase and lowercase vowels from the original string.
 * 2. Iterating through the result, converting each consonant to lowercase and
 *    prepending a "." to it.
 *
 * @warning
 * FLAWED IMPLEMENTATION: This code contains undefined behavior and logical issues.
 * 1.  **Undefined Behavior**: It uses `strlen` on `charRemove`, which is a character
 *     array, not a null-terminated C-string. `strlen` will read past the end
 *     of the array until it happens to find a null byte in memory, leading to
 *     unpredictable behavior. The correct way to get the size would be `sizeof`.
 * 2.  **Logic Error**: The problem requires converting the entire string to
 *     lowercase *before* processing. This implementation omits that step and
 *     instead tries to remove both 'a' and 'A', etc. separately. While it may
 *     pass some test cases, it doesn't follow the problem's explicit rules.
 */
#include <algorithm>
#include <iostream>
#include <string.h>
#include <string>
using namespace std;

int main() {
  string str, s2;
  cin >> str;
  // A list of all vowel characters, both upper and lower case.
  char charRemove[]{'A', 'O', 'Y', 'E', 'U', 'I', 'a', 'o', 'y', 'e', 'u', 'i'};
  // FLAW: Using strlen on a character array that is not a null-terminated string.
  // This is undefined behavior.
  long n = strlen(charRemove);
  /**
   * @brief This loop attempts to remove all vowels from the string.
   * It iterates through each vowel in `charRemove` and applies the erase-remove
   * idiom to the input string `str`.
   */
  for (int j = 0; j < n; ++j) {
    str.erase(remove(str.begin(), str.end(), charRemove[j]), str.end());
  }
  long long len = str.length();
  /**
   * @brief This loop iterates through the remaining characters (consonants),
   * converts them to lowercase, and prepends a '.' to each.
   */
  for (int j = 0; j < len; ++j) {
    s2 += '.';
    s2 += tolower(str[j]);
  }
  cout << s2 << endl;
  ; // Unnecessary empty statement.
  return 0;
}