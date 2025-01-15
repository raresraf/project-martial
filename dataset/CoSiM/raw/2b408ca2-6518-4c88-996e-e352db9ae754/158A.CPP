/*
 * nextround.cpp
 * http://codeforces.com/problemset/problem/158/A
 *  Created on: Oct 22, 2017
 *      Author: athena
 * 31608946 	GNU C++14 	Accepted 	60 ms 	1880 KB
 * 2017-10-22 19:13:36 	2017-10-22 19:13:36
 */

#include <iostream>
#include <vector>
using namespace std;

int main() {
  int n, k, a;
  cin >> n >> k;

  vector<int> participants;

  for (int i = 0; i < n; i++) {
    cin >> a;
    participants.push_back(a);
  }

  int count = 0;

  for (int j = 0; j < n; j++) {
    if (participants[j] >= participants[k - 1] && participants[j] > 0) {
      count++;
    }
  }

  cout << count << endl;

  return 0;
}
