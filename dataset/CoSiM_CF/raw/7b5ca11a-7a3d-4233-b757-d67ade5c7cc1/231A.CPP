#include <bits/stdc++.h>
using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  freopen("input", "r", stdin);
  freopen("output", "w", stdout);
#endif

  int n, minicount, count, a;
  cin >> n;
  count = 0;
  for (int i = 0; i < n; i++) {
    minicount = 0;
    for (int j = 0; j < 3; j++) {
      cin >> a;
      minicount += a;
    }
    if (minicount > 1)
      count++;
  }
  cout << count << endl;
}
