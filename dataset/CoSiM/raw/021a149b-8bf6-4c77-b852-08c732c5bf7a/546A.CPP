/*
 * 546A - Soldier and Bananas.cpp
 * http://codeforces.com/problemset/problem/546/A
 *  Created on: Oct 28, 2017
 *      Author: athena
 * 31848244 	546A - 23 	GNU C++14 	Accepted 	15 ms 	4 KB
 * 2017-10-28 21:46:27 	2017-10-28 21:46:27
 */

#include <iostream>
using namespace std;

int main() {
  int k, n, w;
  cin >> k >> n >> w;

  int totalCost = 0;
  for (int i = 1; i <= w; i++) {
    totalCost += i * k;
  }

  if (totalCost <= n) {
    cout << "0" << endl;
  } else {
    int amountToBorrow = totalCost - n;
    cout << amountToBorrow << endl;
  }

  return 0;
}
