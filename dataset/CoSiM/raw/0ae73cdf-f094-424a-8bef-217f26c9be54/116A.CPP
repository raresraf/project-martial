// 116A.cpp
// A. Tram
// Author: Ikki
// May 31, 2015

#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;
  int val = 0, ans = 0;
  for (int i = 0; i < n; i++) {
    int a, b;
    cin >> a >> b;
    val -= a;
    val += b;
    ans = max(ans, val);
  }
  cout << ans << endl;
  return 0;
}