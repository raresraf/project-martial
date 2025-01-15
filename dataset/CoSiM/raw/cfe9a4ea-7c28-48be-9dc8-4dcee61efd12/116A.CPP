#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  int a, b;
  int maxn = 0;
  int total = 0;
  for (int i = 0; i <= n; i++) {
    cin >> a >> b;
    total -= a;
    total += b;
    maxn = max(total, maxn);
  }
  cout << maxn << endl;
  return 0;
}
