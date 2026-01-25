/*
 *Link: https://codeforces.com/problemset/problem/546/A
 */

#include <iostream>
using namespace std;

int main() {
  long long k, n, w, total_price, res;
  cin >> k >> n >> w;
  total_price = k * ((w * (w + 1)) / 2);
  res = total_price - n;
  if (res <= 0) {
    res = 0;
  }
  cout << res << endl;
}