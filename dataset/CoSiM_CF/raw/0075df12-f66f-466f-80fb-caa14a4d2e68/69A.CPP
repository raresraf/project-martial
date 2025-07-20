/*
 * 69A - Young Physicist.cpp
 * http://codeforces.com/problemset/problem/69/A
 *  Created on: Oct 28, 2017
 *      Author: athena
 * 31849100 	69A - 23 	GNU C++14 	Accepted 	62 ms 	8 KB
 * 2017-10-28 22:26:28 	2017-10-28 22:26:28
 */

#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;

  int x, y, z;
  int countX(0), countY(0), countZ(0);

  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    countX += x;
    countY += y;
    countZ += z;
  }

  if (countX == 0 && countY == 0 && countZ == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
