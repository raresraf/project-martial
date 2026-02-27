/*
 * wordcapitalization.cpp
 * http://codeforces.com/problemset/problem/281/A
 *  Created on: Oct 26, 2017
 *      Author: athena
 * 31761868 	GNU C++14 	Accepted 	30 ms 	8 KB 	2017-10-26
 * 19:22:42 	2017-10-26 19:22:43
 */

/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem from Codeforces.
 * @details The task is to capitalize the first letter of a given word. The
 * rest of the word should remain unchanged.
 *
 * @algorithm
 * 1. Read the input word into a string.
 * 2. Access the first character of the string (at index 0).
 * 3. Convert this character to its uppercase equivalent using `toupper`.
 * 4. Assign the uppercase character back to the first position in the string.
 * 5. Print the modified string.
 *
 * @complexity
 * Time Complexity: O(L) to read the string of length L. The capitalization
 * itself is an O(1) operation.
 * Space Complexity: O(L) to store the string.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  string s;
  cin >> s;

  // Access the first character, convert it to uppercase, and update the string in-place.
  s[0] = toupper(s[0]);

  cout << s << endl;

  return 0;
}
