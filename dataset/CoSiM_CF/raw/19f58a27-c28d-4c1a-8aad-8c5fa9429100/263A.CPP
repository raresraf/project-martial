/*
 * 263A - Beautiful Matrix.cpp
 * http://codeforces.com/problemset/problem/263/A
 *  Created on: Oct 27, 2017
 *      Author: athena
 * 31810135 	263A - 8 	GNU C++14 	Accepted 	30 ms 	4 KB
 * 2017-10-27 18:53:02 	2017-10-27 18:53:02
 */

#include <cmath>
#include <iostream>
using namespace std;

int main() {
  int x = 0;
  int result = 0;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> x;
      if (x == 1) {
        result = abs(i - 2) + abs(j - 2);
      }
    }
  }

  cout << result << endl;

  return 0;
}
