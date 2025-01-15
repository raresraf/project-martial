/**
 * Author     : Jubayer Abdullah Joy
 *              CSE, 5th Batch, University of Barishal
 * Mail       : joy.cse5.bu@gmail.com or jubayerjoy98@gmail.com
 * Codeforces : Joy
 * Problem    : 4A Watermelon
 *
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int testcase;
  cin >> testcase;
  while (testcase--) {
    string word;
    cin >> word;

    if (word.size() <= 10)
      cout << word << '\n';
    else
      cout << word[0] << word.size() - 2 << word[word.size() - 1];
  }

  return 0;
}