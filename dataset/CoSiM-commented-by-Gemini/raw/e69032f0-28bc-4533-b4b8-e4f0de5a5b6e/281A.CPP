/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem from Codeforces.
 * @details This program reads a single word and capitalizes its first letter
 * without changing the rest of the word.
 *
 * Problem description: https://codeforces.com/problemset/problem/281/A
 *
 * @section
 * Algorithm:
 * 1. Read the input word as a string.
 * 2. Access the first character of the string (at index 0).
 * 3. Convert this character to its uppercase equivalent.
 * 4. Print the modified string.
 *
 * @section
 * Complexity Analysis:
 * - Time Complexity: O(N) due to reading and writing the string of length N. The capitalization itself is an O(1) operation.
 * - Space Complexity: O(N) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string a;
  // Read the input word from standard input.
  cin >> a;

  // Capitalize the first letter of the word in-place.
  a[0] = toupper(a[0]);

  // Print the resulting word.
  cout << a;
  
  return 0;
}
