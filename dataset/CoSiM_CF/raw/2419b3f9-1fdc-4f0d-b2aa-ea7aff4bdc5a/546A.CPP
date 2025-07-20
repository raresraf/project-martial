#include <bits/stdc++.h>
using namespace std;
int main() {
  int k, n, w, cnt = 0, cost = 0, dif;
  cin >> k >> n >> w;
  while (cnt != w) {
    cnt++;
    cost += k * cnt;
  }
  dif = cost - n;
  if (dif > 0)
    cout << dif << endl;
  else
    cout << 0 << endl;
  return 0;
}
