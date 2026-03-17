/**
 * @file 281A.CPP
 * @author Nikhil Nagrale
 * @brief Solution to a competitive programming problem, likely Codeforces 281A "Word Capitalization".
 *
 * @details
 * This program reads a single word and capitalizes its first letter. If the first
 * letter is already capitalized, it remains unchanged. The rest of the word is not
 * modified.
 *
 * @algorithm
 * The algorithm is straightforward:
 * 1. Read the entire word into a string.
 * 2. Access the first character of the string at index 0.
 * 3. Use the `toupper()` function to convert this character to its uppercase
 *    equivalent.
 * 4. The modification is done in-place.
 * 5. Print the modified string.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the length of the string, dominated by the
 *   input/output operations.
 * - Space Complexity: O(N), to store the input string.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function and entry point of the program.
 */
int main() {
  string s;
  cin >> s;
  
  // Access the first character and convert it to uppercase in-place.
  s[0] = toupper(s[0]);
  
  // Print the resulting string.
  cout << s << endl;
  
  return 0;
}
