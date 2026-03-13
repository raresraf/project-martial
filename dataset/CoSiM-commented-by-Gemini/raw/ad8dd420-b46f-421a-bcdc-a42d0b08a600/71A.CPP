/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces.
 *
 * This program reads a series of words and shortens any word that is longer
 * than 10 characters. The shortened word is represented by its first character,
 * the number of characters between the first and last character, and its last
 * character. This implementation reads all words into a vector before
 * processing.
 *
 * @see https://codeforces.com/problemset/problem/71/A
 */
//
//  71A.cpp
//  Codeforces
//
//  Created by Tico on 3/17/17.
//  Copyright   2017 Thiago Ribeiro. All rights reserved.
//

#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
  int n;
  vector<string> words;
  string str;

  cin >> n;

  /**
   * @brief Reads all input words into a vector.
   */
  while (n--) {
    cin >> str;
    words.push_back(str);
  }

  /**
   * @brief Processes and prints the words.
   *
   * This loop iterates through the vector of words. If a word's length is
   * greater than 10, it prints the shortened version; otherwise, it prints
   * the original word.
   */
  for (const string s : words) {
    if (s.length() > 10) {
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    } else
      cout << s << endl;
  }

  return 0;
}
