#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  short n, a, b, c, cnt = 0;

  cin >> n;
  while (n--) {
    cin >> a >> b >> c;
    if ((a && b) || (b && c) || (c && a))
      cnt++;
  }
  cout << cnt << "\n";
}