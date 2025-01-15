#include <bits/stdc++.h>
using namespace std;
#define fastio                                                                 \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);

int main() {
  fastio int a = 0, b = 0, c = 0;
  int n, x, y, z;
  cin >> n;
  while (n--) {
    cin >> x >> y >> z;
    a += x, b += y, c += z;
  }
  cout << (a || b || c ? "NO" : "YES");
}
