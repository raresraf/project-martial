/*
 * stonesonthetable.cpp
 * http://codeforces.com/problemset/problem/266/A
 *  Created on: Oct 25, 2017
 *      Author: athena
 * 31720858 	GNU C++14 	Accepted 	30 ms 	4 KB 	2017-10-25
 * 21:37:42 	2017-10-25 21:37:42
 */

#include <iostream>
#include <string>
using namespace std;

int main() {
  int n;
  string stones;
  cin >> n >> stones;

  int counter = 0;

  for (int i = 0; i < n; i++) {
    if (stones[i] == stones[i - 1]) {
      counter++;
    }
  }

  cout << counter << endl;

  return 0;
}
