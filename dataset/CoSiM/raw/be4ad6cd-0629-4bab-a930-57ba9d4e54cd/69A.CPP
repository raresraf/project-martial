#include <bits/stdc++.h>
#define fastIO                                                                 \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0);

using namespace std;

int main() {
  fastIO int n;
  cin >> n;
  int x[n], y[n], z[n];
  int a = 0, b = 0, c = 0;
  for (int i = 0; i < n; i++) {
    cin >> x[i] >> y[i] >> z[i];
  }
  for (int i = 0; i < n; i++) {
    a += x[i];
    b += y[i];
    c += z[i];
  }
  if (a == 0 && b == 0 && c == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}
