/*
 * wordcapitalization.cpp
 * http://codeforces.com/problemset/problem/281/A
 *  Created on: Oct 26, 2017
 *      Author: athena
 * 31761868 	GNU C++14 	Accepted 	30 ms 	8 KB 	2017-10-26
 * 19:22:42 	2017-10-26 19:22:43
 */

#include <iostream>
#include <string>
using namespace std;

int main() {
  string s;
  cin >> s;

  s[0] = toupper(s[0]);

  cout << s << endl;

  return 0;
}
