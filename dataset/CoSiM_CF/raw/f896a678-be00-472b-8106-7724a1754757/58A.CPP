/*
 * 58A - Chat Room.cpp
 * http://codeforces.com/problemset/problem/58/A
 *  Created on: Oct 27, 2017
 *      Author: athena
 * 31815122 	58A - 10 	GNU C++14 	Accepted 	15 ms 	4 KB
 * 2017-10-27 21:00:28 	2017-10-27 21:00:28
 */

#include <iostream>
using namespace std;

int main() {
  string s;
  cin >> s;

  string hello = "hello";
  int j = 0;
  int count = 0;

  for (int i = 0; i < s.size(); i++) {
    if (s[i] == hello[j]) {
      count++;
      j++;
    }
  }

  if (count == 5) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
