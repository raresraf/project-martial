/**
 * @file 281A.CPP
 * @brief Solution for Codeforces problem 281A, "Word Capitalization".
 *
 * @description
 * This program capitalizes the first letter of a given word. It uses the
 * standard C++ library function `islower` to check the case of the first
 * character before applying the transformation.
 *
 * @algorithm
 * 1. Read the input word into a std::string.
 * 2. Use `islower()` to check if the first character is a lowercase letter.
 * 3. If it is, convert it to its uppercase equivalent by subtracting 32 from
 *    its ASCII value.
 * 4. Print the potentially modified string.
 *
 * @time_complexity O(L), where L is the length of the string, dominated by I/O.
 * @space_complexity O(L) to store the string.
 */
#include <cctype>
#include <iostream>
#include <string>
using namespace std;

int main() {
  string s;
  cin >> s;

  // Block Logic: Check if the first character is lowercase and convert if necessary.
  // The islower() function provides a standard way to check character case.
  if (islower(s[0])) {
    // Convert to uppercase by subtracting 32 from the ASCII value.
    // An alternative standard C++ approach would be `s[0] = toupper(s[0]);`
    s[0] = s[0] - 32;
  }

  cout << s;
}
