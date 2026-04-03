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
 *    a. Read a word.
 *    b. If the word's length is greater than 10, print the abbreviation.
 *    c. Otherwise, print the word.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is the length of the longest word (for reading the input). The processing for each word is O(1).
 * Space Complexity: O(L) for storing the longest word.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Processes a single word, printing either the word itself or its abbreviation.
 */
void change() {
  string s;
  cin >> s;
  int size = s.size();
  
  // If the word's length is greater than 10, abbreviate it.
  if (size > 10) {
    cout << s[0] << size - 2 << s[size - 1] << endl;
  } else {
    // Otherwise, print the word as is.
    cout << s << endl;
  }
}

int main() {
  // Read the number of words to process.
  int n;
  cin >> n;
  
  // Loop n times to process each word.
  while (n--) {
    change();
  }
}
