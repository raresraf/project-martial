#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  int res = 0;
  int p, v, t;
  while (n--) {
    cin >> p >> v >> t;
    if (p + v + t >= 2)
      res++;
  }
  cout << res;
}
