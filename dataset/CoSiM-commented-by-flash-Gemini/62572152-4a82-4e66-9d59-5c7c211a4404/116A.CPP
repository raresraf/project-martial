#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  int a, b;
  int res = 0, max = -1;
  while (n--) {
    cin >> a >> b;
    if (a == 0)
      res = res + b;
    else
      res = res - a + b;
    if (res > max)
      max = res;
  }
  cout << max;
}
