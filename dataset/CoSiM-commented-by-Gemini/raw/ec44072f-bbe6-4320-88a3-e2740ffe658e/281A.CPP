/**
 * @file 281A.CPP
 * @brief Capitalizes the first letter of a given word.
 * 
 * This program reads a single word from standard input, converts its first
 * character to uppercase, and then prints the modified word. It's a basic
 * string manipulation task.
 * 
 * @b Algorithm: Direct character modification.
 * @b Time_Complexity: O(L) where L is the length of the input string (dominated by I/O). The capitalization is O(1).
 * @b Space_Complexity: O(L) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  // Ensure the first character of the string is uppercase.
  s[0] = toupper(s[0]);
  cout << s;
}
