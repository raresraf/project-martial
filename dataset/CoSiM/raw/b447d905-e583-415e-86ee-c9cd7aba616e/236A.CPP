/*
 * 236A - Boy or Girl.cpp
 * http://codeforces.com/problemset/problem/236/A
 *  Created on: Oct 27, 2017
 *      Author: athena
 * 31817469 	236A - 10 	GNU C++14 	Accepted 	30 ms 	8 KB
 * 2017-10-27 22:25:52 	2017-10-27 22:25:52
 */

#include <algorithm>
#include <iostream>
#include <string>
using namespace std;

int main() {
  string s;
  cin >> s;

  sort(s.begin(), s.end());
  s.erase(unique(s.begin(), s.end()), s.end());

  if (s.length() % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
  } else {
    cout << "IGNORE HIM!" << endl;
  }

  return 0;
}
