/*
 * petyaandstrings.cpp
 * http://codeforces.com/problemset/problem/112/A
 *  Created on: Oct 24, 2017
 *      Author: athena
 * 31662589 	GNU C++14 	Accepted 	30 ms 	1872 KB
 * 2017-10-23 23:51:50 	2017-10-23 23:51:50
 */

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string str1, str2;
  cin >> str1 >> str2;

  transform(str1.begin(), str1.end(), str1.begin(), ::tolower);
  transform(str2.begin(), str2.end(), str2.begin(), ::tolower);

  strcmp(str1.c_str(), str2.c_str());

  if (str1 < str2) {
    cout << "-1" << endl;
  } else if (str1 > str2) {
    cout << "1" << endl;
  } else {
    cout << "0" << endl;
  }

  return 0;
}
