/*
 * 110A - Nearly Lucky Number.cpp
 * http://codeforces.com/problemset/problem/110/A
 *  Created on: Oct 28, 2017
 *      Author: athena
 * 31849461 	110A - 20 	GNU C++14 	Accepted 	30 ms 	4 KB
 * 2017-10-28 22:44:58 	2017-10-28 22:44:58
 */

#include <iostream>
#include <string>
using namespace std;

int main() {
  string s;
  cin >> s;

  int counter = 0;

  for (int i = 0; i < s.size(); i++)
    if (s[i] == '4' || s[i] == '7') {
      counter++;
    }

  if (counter == 4 || counter == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
