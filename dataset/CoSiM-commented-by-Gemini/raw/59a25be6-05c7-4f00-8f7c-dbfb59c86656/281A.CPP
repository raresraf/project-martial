/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem from Codeforces (281A).
 * @details The problem requires capitalizing the first letter of a given word.
 * This solution leverages the standard C++ library for a concise implementation.
 *
 * @algorithm
 * The solution reads the input word as a C++ string. It then directly applies the
 * `toupper()` function from the `<cctype>` library to the first character of the
 * string (`s[0]`). This function handles the conversion from lowercase to uppercase
 * and correctly does nothing if the character is already uppercase or not a letter.
 * The modified string is then printed to the standard output.
 *
 * @complexity
 * Time Complexity: O(L) where L is the length of the string, as this is dominated
 * by the I/O operations. The core capitalization logic is O(1).
 * Space Complexity: O(L) to store the input string.
 */
/*
 *	User: Isanchez_Aguilar
 *	Problem: CodeForces 281A - Word Capitalization
 */
#include <bits/stdc++.h>

using namespace std;

int main(void) {
  /**
   * Functional Utility: Decouples C++ streams from standard C streams and
   * unties cin from cout to optimize I/O performance, which is a common
   * practice in competitive programming.
   */
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);

  /**
   * @variable s: Stores the input word.
   */
  string s;
  cin >> s;

  // Capitalize the first character of the string using the standard `toupper` function.
  // This is the most idiomatic and safe way to perform case conversion.
  s[0] = toupper(s[0]);

  // Output the modified string.
  cout << s << "\n";

  return 0;
}