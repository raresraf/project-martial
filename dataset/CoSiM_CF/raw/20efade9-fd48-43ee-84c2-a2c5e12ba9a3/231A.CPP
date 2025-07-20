/*
 * team.cpp
 * http://codeforces.com/problemset/problem/231/A
 *  Created on: Oct 23, 2017
 *      Author: athena
 * 31658165 	GNU C++14 	Accepted 	30 ms 	1876 KB
 * 2017-10-23 21:26:36 	2017-10-23 21:45:23
 */

#include <iostream>
using namespace std;

int main() {
  int n, p, v, t;
  cin >> n;

  int count = 0;

  for (int i = 0; i < n; i++) {
    cin >> p >> v >> t;
    if (p + v + t > 1) {
      count++;
    }
  }

  cout << count << endl;

  return 0;
}
