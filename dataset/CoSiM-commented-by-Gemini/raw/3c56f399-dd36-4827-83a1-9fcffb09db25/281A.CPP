/**
 * @file 281A.CPP
 * @brief Solution to the "Word Capitalization" problem (Codeforces 281A).
 * 
 * @author SHEHANUK AHSAN
 * 
 * @algorithm
 * The problem requires capitalizing the first letter of a given word.
 * 1. Read the input word into a character array.
 * 2. Access the first character of the array (at index 0).
 * 3. Convert this character to its uppercase equivalent using the `toupper()` function.
 * 4. Print the modified character array.
 *
 * @complexity
 * - Time Complexity: O(L), where L is the length of the word, for reading and
 *   printing the string. The capitalization itself is an O(1) operation.
 * - Space Complexity: O(1000) due to the fixed-size character array, which is
 *   effectively O(1) as it doesn't scale with the input size beyond the limit.
 */

// made by : SHEHANUK AHSAN
//           3rd batch, Dept of CSE
//           BAUAT;

#include <cctype>
#include <cstdio>
#include <iostream>
#include <string.h>
using namespace std;

/**
 * @note The use of `main()` without a return type is non-standard in C++.
 * Modern C++ requires `int main()`.
 */
main() {
  // These variables are declared but never used in the program.
  int n, i, j, k;

  // A fixed-size C-style character array is used to store the word.
  // This is potentially unsafe; std::string would be a better choice to avoid
  // buffer overflows.
  char a[1000];

  // Read the word from standard input.
  cin >> a;

  // Capitalize the first letter of the word.
  a[0] = toupper(a[0]);

  // Print the resulting word.
  cout << a;
  return 0;
}