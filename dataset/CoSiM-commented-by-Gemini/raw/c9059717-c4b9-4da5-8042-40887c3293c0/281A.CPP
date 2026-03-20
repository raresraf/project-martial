/**
 * @file 281A.CPP
 * @brief Solution for Codeforces problem 281A, "Word Capitalization".
 * @details This program capitalizes the first letter of a given word. This
 * implementation uses a distinct method of reading the first character separately
 * from the rest of the string.
 *
 * @section Algorithm
 * The approach taken is to treat the first character as a special case from the outset.
 * 1. The program first reads a single character using `scanf` into the `ch` variable.
 *    This captures the first letter of the word.
 * 2. It then reads the remainder of the word into a `std::string` `s` using `cin`.
 * 3. It checks if the first character `ch` is a lowercase letter.
 * 4. If it is, the character is converted to uppercase using ASCII arithmetic
 *    (`ch - 'a' + 'A'`) and printed.
 * 5. If it's not a lowercase letter, it's printed without modification.
 * 6. Finally, the rest of the string `s` is printed.
 *
 * @subsection Complexity
 * - Time Complexity: O(N), where N is the length of the input word. This is
 *   dominated by the I/O operations.
 * - Space Complexity: O(N), for storing the part of the string after the first character.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // The `freopen` line is commented out but is typically used for local testing.
  // freopen("input.txt", "r", stdin);
  
  // `ch` will store the first character of the input.
  char ch;
  // `s` will store the rest of the input word.
  string s;
  
  // Read only the first character from the input stream.
  scanf("%c", &ch);
  // Read the rest of the word (from the second character) into the string `s`.
  cin >> s;
  
  /**
   * @brief This conditional block processes the first character.
   * It checks if the character is lowercase and converts it to uppercase if so.
   */
  if (ch >= 'a' && ch <= 'z')
    // Manually convert to uppercase using ASCII value manipulation.
    printf("%c", ch - 'a' + 'A');
  else
    // If not a lowercase letter, print it unchanged.
    printf("%c", ch);
  
  // Print the rest of the string that was read into `s`.
  cout << s << endl;
  return 0;
}
