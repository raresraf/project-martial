#include <bits/stdc++.h>
using namespace std;

#define IOS                                                                    \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define endl "\n"
#define int long long

const int N = 1e5 + 5;

int32_t main() {
  IOS;
  string a, b;
  cin >> a >> b;

  for (int i = 0; i < a.length(); i++) {
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);

    if (a[i] > b[i]) {
      cout << 1 << endl;
      return 0;
    }

    else if (a[i] < b[i]) {
      cout << -1 << endl;
      return 0;
    }
  }

  cout << 0 << endl;

  return 0;
}
