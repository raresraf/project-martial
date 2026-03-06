/**
 * @file 281A.CPP
 * @brief Capitalizes the first letter of a given word.
 *
 * This program reads a single word and ensures its first letter is uppercase.
 * All other letters in the word remain unchanged.
 *
 * @algorithm In-place character modification.
 * @complexity
 * Time: O(L) to read the string of length L. The capitalization itself is an O(1)
 *       operation as it only affects the first character.
 * Space: O(L) to store the input string.
 */
#include <iostream>
#include <string>
#include <cctype> // For toupper()

// The <bits/stdc++.h> header is a non-standard GNU extension that includes
// most of the standard library headers. It's common in competitive programming.
// using namespace std; is also a common practice in this context for brevity.

using namespace std;

/**
 * @brief Main function to handle input and capitalization.
 * @note The use of 'main()' without an explicit 'int' return type is a
 *       non-standard C++ extension (in C it defaults to int, but not in C++).
 *       Standard C++ requires 'int main()'.
 */
int main() {
  // s: stores the input word.
  string s;
  cin >> s;
  
  // Access the first character of the string and convert it to uppercase.
  // If the character is already uppercase or not a letter, toupper has no effect.
  s[0] = toupper(s[0]);
  
  // Print the modified string.
  cout << s;
  
  // According to the C++ standard, main should return an integer.
  return 0;
}
