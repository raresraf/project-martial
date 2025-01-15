/*
 * 266B - Queue at the School.cpp
 * http://codeforces.com/problemset/problem/266/B
 *  Created on: Nov 2, 2017
 *      Author: athena
 * 31989819 	266B - 10 	GNU C++14 	Accepted 	30 ms 	4 KB
 * 2017-11-02 21:45:03 	2017-11-02 21:45:03
 */

#include <iostream>
using namespace std;

int main() {
  int n, t;
  string s;
  cin >> n >> t >> s;

  while (t--) {
    for (int i = 0; i < n; i++) {
      if (s[i] == 'G' && s[i - 1] == 'B') {
        s[i] = 'B';
        s[i - 1] = 'G';
        i++;
      }
    }
  }

  cout << s << endl;

  return 0;
}
