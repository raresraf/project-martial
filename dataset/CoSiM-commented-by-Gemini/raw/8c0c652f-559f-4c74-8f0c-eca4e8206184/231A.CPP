// 231A.cpp
// A. Team
// Author: Ikki
// May 31, 2015

#include <iostream>
using namespace std;

int main() {
  int n, ans = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    int a, b, c;
    cin >> a >> b >> c;
    if (a + b + c >= 2)
      ans++;
  }
  cout << ans << endl;
  return 0;
}