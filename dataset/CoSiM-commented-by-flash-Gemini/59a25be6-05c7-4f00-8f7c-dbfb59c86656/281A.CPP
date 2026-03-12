/*
 *	User: Isanchez_Aguilar
 *	Problem: CodeForces 281A - Word Capitalization
 */
/**
 * @59a25be6-05c7-4f00-8f7c-dbfb59c86656/281A.CPP
 * @brief This program solves the "Word Capitalization" problem (Codeforces 281A).
 * It reads a single word from standard input, capitalizes its first letter,
 * and then prints the modified word to standard output.
 *
 * Algorithm: Direct string manipulation - specifically, in-place modification of the first character.
 * Time Complexity: O(L) where L is the length of the input word. Effectively O(1) for typical word lengths.
 * Space Complexity: O(L) for storing the string. Effectively O(1) for typical word lengths.
 */
#include <bits/stdc++.h> // Includes most standard C++ library headers.

using namespace std; // Uses the standard namespace.

int main(void) {
  // Block Logic: Optimizes C++ standard I/O operations.
  // `ios_base::sync_with_stdio(0)` unties C++ streams from C stdio.
  // `cin.tie(0)` unties cin from cout, preventing flushing of cout before cin operations.
  // `cout.tie(0)` further unties cout from cin, potentially speeding up output.
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);

  string s; // Declares a string variable to store the input word.
  cin >> s; // Reads the word from standard input.

  // Block Logic: Capitalizes the first letter of the input word.
  // Pre-condition: The string 's' is not empty.
  // Invariant: The first character of 's' is converted to its uppercase equivalent.
  // Inline: `toupper()` converts a character to its uppercase form if it's a lowercase letter;
  //         otherwise, it returns the character unchanged.
  s[0] = toupper(s[0]); 

  cout << s << "\n"; // Prints the modified string, followed by a newline.

  return 0; // Indicates successful program termination.
}