//
// Created by Taro Young on 2020/9/7.
//
#include <iostream>

using namespace std;

int main() {
  int k, n, w;
  cin >> k >> n >> w;
  int money = w * (w + 1) / 2 * k;
  if (money > n)
    cout << money - n;
  else
    cout << 0;
  return 0;
}