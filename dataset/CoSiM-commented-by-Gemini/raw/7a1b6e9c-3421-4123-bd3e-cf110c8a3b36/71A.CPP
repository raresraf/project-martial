/*
 * waytoolongwords.cpp
 * http://codeforces.com/problemset/problem/71/A
 *  Created on: Oct 21, 2017
 *     Author: athena
 * 31582116 	GNU C++14 	Accepted 	30 ms 	1876 KB
 * 2017-10-21 20:24:32 	2017-10-21 20:24:32
 */

#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;

  string str;

  for (int i = 0; i < n; i++) {
    cin >> str;

    if (str.size() <= 10) {
      cout << str << endl;
    } else {
      cout << str[0] << str.size() - 2 << str[str.size() - 1] << endl;
    }
  }

  return 0;
}
