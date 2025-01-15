/*
 * 160A - Twins.cpp
 * http://codeforces.com/problemset/problem/160/A
 *  Created on: Oct 27, 2017
 *      Author: athena
 * 31813150 	160A - 20 	GNU C++14 	Accepted 	62 ms 	8 KB
 * 2017-10-27 19:56:03 	2017-10-27 19:56:03
 */

#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

int main() {
  int n;
  cin >> n;

  vector<int> coins;
  int coin;
  int amount = 0;

  for (int i = 0; i < n; i++) {
    cin >> coin;
    coins.push_back(coin);
    amount += coin;
  }

  sort(coins.begin(), coins.end(), greater<int>());

  int numberOfCoins = 0;
  int myAmount = 0;

  for (int j = 0; j < coins.size(); j++) {
    numberOfCoins++;
    myAmount += coins[j];
    if (myAmount > amount - myAmount) {
      break;
    }
  }

  cout << numberOfCoins << endl;

  return 0;
}
