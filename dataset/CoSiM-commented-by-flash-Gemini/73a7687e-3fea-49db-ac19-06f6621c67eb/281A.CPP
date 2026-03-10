/**
 * @file 281A.CPP
 * @brief This program capitalizes the first letter of an input word (string).
 *   It checks if the first character is a lowercase letter and converts it to uppercase if needed.
 *   Problem description: https://codeforces.com/problemset/problem/281/A
 * * Algorithm: Character case conversion using ASCII values.
 * Time Complexity: O(L) where L is the length of the input string, primarily due to string I/O.
 * Space Complexity: O(L) to store the input string.
 */

/**
* Git >> Mohsin_Riad
**/
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Macro to optimize C++ standard I/O operations.
 * Functional Utility: Decouples C++ streams from standard C streams and unties cin from cout,
 *                     which can significantly speed up I/O in competitive programming.
 */
#define IOS                                                                    \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define mx 1000 // Defines a constant for maximum size, possibly for arrays.
#define endl "\n" // Defines 'endl' as a newline character for faster output than std::endl (which also flushes).
#define pb push_back // Alias for std::vector::push_back.
#define pob pop_back // Alias for std::vector::pop_back.
typedef long long ll; // Typedef for long long integer.
typedef double dd;   // Typedef for double.
const int N = 1e6; // Defines a constant N, possibly for array sizes or loop limits.

/**
 * @brief Main function to execute the string capitalization logic.
 * Functional Utility: Configures I/O, reads a string, capitalizes its first letter if necessary, and prints the result.
 */
int main() {
  // Functional Utility: Applies I/O optimizations for faster input and output.
  IOS

      char str[1005]; // Declares a character array (C-style string) to store the input.
  // Block Logic: Reads the input string into the 'str' character array.
  cin >> str;

  /**
   * Block Logic: Checks if the first character of the string is a lowercase letter (ASCII 'a' to 'z').
   * If it is, it converts it to an uppercase letter by subtracting 32 from its ASCII value.
   * Invariant: After this block, the first character of 'str' will be uppercase if it was originally a lowercase letter.
   */
  if (str[0] >= 97 && 122 >= str[0]) // ASCII values: 'a' = 97, 'z' = 122
    str[0] -= 32;                   // ASCII difference between lowercase and uppercase is 32

  // Outputs the modified string followed by a newline.
  cout << str << endl;
  return (0);
}
