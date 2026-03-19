/**
 * @file 281A.CPP
 * @brief Solution for the "Word Capitalization" problem (Codeforces 281A).
 * @details This program takes a single word as input and capitalizes its first letter.
 *
 * Algorithm: Conditional Capitalization
 * The program reads the input word into a std::string. It then checks if the first
 * character is lowercase using the `islower()` function. If it is, the character
 * is converted to its uppercase equivalent with `toupper()`. Finally, the
 * potentially modified string is printed.
 *
 * Time Complexity: O(L), where L is the length of the word, dominated by I/O.
 * The capitalization logic itself is O(1).
 * Space Complexity: O(L) to store the input string.
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
using namespace std;

int main() {
  // s: stores the input word. Initialized with a space but immediately overwritten.
  string s = " ";
  cin >> s;

  // Core Logic: Check if the first character is lowercase.
  if (islower(s[0]))
    // If it is, convert it to uppercase.
    s[0] = toupper(s[0]);
  
  // Print the resulting string.
  cout << s << endl;

  return 0;
}
