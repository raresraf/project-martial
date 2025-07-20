/*
 * bitplusplus.cpp
 * http://codeforces.com/problemset/problem/282/A
 *  Created on: Oct 23, 2017
 *      Author: athena
 * http://codeforces.com/problemset/problem/282/A
 * 31660146 	GNU C++14 	Accepted 	15 ms 	1876 KB
 * 2017-10-23 22:19:50 	2017-10-23 22:19:50
 */

#include <iostream>
#include <string>
using namespace std;

int main() {
  int n;
  cin >> n;

  int x = 0;
  string operation;

  for (int i = 0; i < n; i++) {
    cin >> operation;
    if (operation[1] == '+') {
      x++;
    } else {
      x--;
    }
  }

  cout << x << endl;

  return 0;
}
