/*
 * watermelon.cpp
 * http://codeforces.com/problemset/problem/4/A
 *  Created on: Oct 21, 2017
 *      Author: athena
 * 31580871 	GNU C++14 	Accepted 	30 ms 	1880 KB
 * 2017-10-21 19:31:33 	2017-10-21 19:31:33
 */

#include <iostream>
using namespace std;

int main() {
  int n = 0;
  cin >> n;

  if (n > 2 && n % 2 == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
