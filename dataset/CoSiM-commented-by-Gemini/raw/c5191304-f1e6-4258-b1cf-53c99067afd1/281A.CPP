/**
 * @file 281A.CPP
 * @brief Solution to a competitive programming problem: Capitalize the first letter of a word.
 *
 * @section Algorithm
 * 1. Read a word from standard input into a character array.
 * 2. Check if the first character is a lowercase English letter.
 * 3. If it is, convert it to its uppercase equivalent by manipulating its ASCII value.
 * 4. Print the resulting word.
 *
 * @section Complexity
 * - Time Complexity: O(L) where L is the length of the input word, dominated by the input operation. The capitalization itself is O(1).
 * - Space Complexity: O(S) where S is the fixed size of the character array (1000).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // Declare a C-style character array to store the word. A fixed size of 1000 is allocated.
  // Using std::string would be more idiomatic and safer in C++.
  char w[1000];
  cin >> w;

  // Store the ASCII value of the first character. This is not strictly necessary but is done here.
  int a = w[0];

  /**
   * Block Logic: Check if the first character is a lowercase letter and, if so, capitalize it.
   * Pre-condition: w[0] contains the first character of the input word.
   */
  if (w[0] >= 'a') {
    // Post-condition: Convert the character to uppercase by subtracting 32 from its ASCII value.
    // This relies on the ASCII encoding where lowercase and uppercase letters are 32 apart.
    // A more portable method would be to use the `toupper()` function from <cctype>.
    w[0] = a - 32;
  }

  // Print the modified word to standard output.
  cout << w << endl;

  return 0;
}
