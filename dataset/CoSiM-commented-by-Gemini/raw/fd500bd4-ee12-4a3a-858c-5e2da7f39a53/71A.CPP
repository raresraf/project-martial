/**
 * @file 71A.cpp
 * @brief Solution to the "Way Too Long Words" problem.
 * This program reads a number of words and, for each word, if its length is
 * greater than 10, it prints an abbreviation of it. Otherwise, it prints the
 * word as is. The abbreviation consists of the first character, the number of
 * characters between the first and last characters, and the last character.
 *
 * Algorithm:
 * 1. Read the number of words `n`.
 * 2. Loop `n` times:
 *    a. Read a word into a string array.
 *    b. If the word's length is greater than 10, print the abbreviation.
 *    c. Otherwise, print the word.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is the length of the longest word.
 * Space Complexity: O(N * L) due to the string array used to store all words.
 */
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Arrays to store the words, though not strictly necessary as each can be processed in place.
string s[123];

int main(int argc, char const *argv[]) {

  int n;
  std::cin >> n;
  // Loop n times to process each word.
  for (int i = 0; i < n; i++) {
    std::cin >> s[i];
    // If the word's length is 10 or less, print it as is.
    if (s[i].size() <= 10) {
      std::cout << s[i] << '\n';
    } else {
      // Otherwise, print the abbreviation.
      std::cout << s[i][0] << s[i].size() - 2 << s[i][s[i].size() - 1] << '\n';
    }
  }

  return 0;
}
