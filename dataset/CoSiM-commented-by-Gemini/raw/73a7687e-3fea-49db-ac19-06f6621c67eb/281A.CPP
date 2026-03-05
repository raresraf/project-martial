/**
 * @file 281A.CPP
 * @brief Capitalizes the first letter of a given input string.
 *
 * This program reads a single word or string from standard input and ensures
 * that its first character is uppercase. If the first character is already
 * uppercase or not a letter, it remains unchanged. Otherwise, it is
 * converted from lowercase to uppercase.
 *
 * @algorithm The core logic involves checking if the first character of the
 * string falls within the ASCII range for lowercase English letters ('a' through 'z').
 * If it does, 32 is subtracted from its ASCII value to convert it to the
 * corresponding uppercase letter.
 *
 * @complexity
 * Time Complexity: O(L) where L is the length of the input string, dominated by input reading. The capitalization logic itself is O(1).
 * Space Complexity: O(S) where S is the maximum string size (1005 characters in this case).
 */

// Includes all standard C++ libraries, a common practice in competitive programming for convenience.
#include <bits/stdc++.h>
using namespace std;

/**
 * @def IOS
 * @brief Macro to optimize I/O operations.
 *
 * This macro unties cin/cout from stdio and disables synchronization with C-style I/O,
 * significantly speeding up input and output operations.
 */
#define IOS                                                                    \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0);

// Defines a constant for maximum size, though it's unused in this specific file.
#define mx 1000
// Redefines the newline character for potentially faster output.
#define endl "\n"
// Defines a shorthand for the push_back vector operation.
#define pb push_back
// Defines a shorthand for the pop_back vector operation.
#define pob pop_back

// Type aliases for common data types.
typedef long long ll;
typedef double dd;
// Defines a large integer constant, likely for array sizes in other problems.
const int N = 1e6;

int main() {
  // Applies the I/O optimization macro.
  IOS

  // Declares a character array to store the input string, with a size of 1005.
  char str[1005];
  // Reads the input string from stdin.
  cin >> str;

  /**
   * @brief Capitalization logic.
   * Checks if the first character is a lowercase letter using its ASCII value.
   * ASCII 'a' is 97 and 'z' is 122.
   */
  if (str[0] >= 97 && 122 >= str[0])
    // Converts the character to uppercase by subtracting 32 from its ASCII value.
    str[0] -= 32;

  // Prints the modified (or original) string to stdout.
  cout << str << endl;
  return (0);
}
