/**
 * @file 281A.CPP
 * @brief This program reads a single word (string) as input and capitalizes its first letter.
 * The modified word, with the first letter in uppercase and the rest unchanged, is then printed.
 * This is a basic string manipulation task, often encountered as an introductory problem.
 * Algorithm: Direct string access and character case conversion.
 * Time Complexity: O(L), where L is the length of the input string, mainly due to reading the string.
 * Space Complexity: O(L) for storing the input string.
 */
// made by : SHEHANUK AHSAN
//           3rd batch, Dept of CSE
//           BAUAT;

#include <cctype>     // Required for `toupper` function.
#include <cstdio>     // Standard input/output functions (e.g., printf, scanf), though not explicitly used here.
#include <iostream>   // Required for `cin`, `cout`.
#include <string.h>   // Required for string manipulation functions (e.g., strlen), though `std::string` is preferred in modern C++.
using namespace std;

// The 'main' function is declared with a non-standard return type.
// In C++, 'main' should return 'int'. The code will still compile but it's not standard compliant.
main() {
  int n, i, j, k; // Declares integer variables, though most are unused in this specific solution.

  char a[1000]; // Declares a character array of size 1000 to store the input string.
  cin >> a; // Reads the input word into the character array `a`.
  /**
   * Functional Utility: Converts the first character of the string `a` to its uppercase equivalent.
   * `toupper` is a standard library function from `<cctype>`.
   */
  a[0] = toupper(a[0]);

  cout << a; // Prints the modified string `a` to standard output.
  return 0; // Indicates successful program execution.
}