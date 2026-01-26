#include <bits/stdc++.h>
using namespace std;
int main() {
  int t, a, b, c, ck = 0;
  scanf("%d", &t);
  while (t--) {
    cin >> a >> b >> c;
    if ((a + b + c) >= 2) {
      ck++;
    }
  }
  cout << ck;
  return 0;
}
