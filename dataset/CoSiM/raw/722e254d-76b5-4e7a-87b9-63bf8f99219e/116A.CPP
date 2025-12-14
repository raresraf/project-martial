/*
 * tram.cpp
 * http://codeforces.com/problemset/problem/116/A
 *  Created on: Oct 26, 2017
 *      Author: athena
 * 31757462 	GNU C++14 	Accepted 	30 ms 	8 KB 	2017-10-26
 * 18:52:15 	2017-10-26 18:52:16
 */

#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;

  int a, b;

  int counter = 0;
  int max = 0;

  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    counter -= a;
    counter += b;
    if (counter > max) {
      max = counter;
    }
  }

  cout << max << endl;

  return 0;
}
