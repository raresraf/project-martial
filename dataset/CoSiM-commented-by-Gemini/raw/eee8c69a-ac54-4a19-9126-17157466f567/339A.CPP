/*
 * helpfulmaths.cpp
 * http://codeforces.com/problemset/problem/339/A
 *  Created on: Oct 25, 2017
 *      Author: athena
 * 31693931 	GNU C++14 	Accepted 	30 ms 	1876 KB
 * 2017-10-24 23:56:22 	2017-10-24 23:56:22
 */

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

vector<int> parseString(string str) {
  stringstream ss(str);
  int integer;
  char plus;
  vector<int> temp;

  while (!ss.eof()) {
    ss >> integer >> plus;
    temp.push_back(integer);
  }

  return temp;
}

int main() {
  string s;
  cin >> s;

  vector<int> summands = parseString(s);

  sort(summands.begin(), summands.end());

  int i;

  for (i = 0; i < summands.size() - 1; i++) {
    cout << summands[i] << "+";
  }
  cout << summands[i] << endl;

  return 0;
}
