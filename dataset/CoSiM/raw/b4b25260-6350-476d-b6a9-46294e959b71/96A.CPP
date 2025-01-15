/*
 * football.cpp
 * http://codeforces.com/problemset/problem/96/A
 *  Created on: Oct 23, 2017
 *      Author: athena
 * 31661421 	GNU C++14 	Accepted 	60 ms 	1876 KB
 * 2017-10-23 23:00:01 	2017-10-23 23:00:01
 */

#include <iostream>
#include <string>
using namespace std;

int main() {
  string str;
  cin >> str;

  int count = 0;

  for (int i = 0; i < str.size(); i++) {
    if (str[i] == str[i - 1]) {
      count++;
      if (count == 6) {
        cout << "YES" << endl;
        return 0;
      }
    } else {
      count = 0;
    }
  }

  cout << "NO" << endl;

  return 0;
}
