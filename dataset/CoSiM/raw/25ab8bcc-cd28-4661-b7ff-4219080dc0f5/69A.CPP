#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main() {
  int n;
  cin >> n;
  int x[n], y[n], z[n];
  for (int i = 0; i < n; i++) {
    cin >> x[i] >> y[i] >> z[i];
  }
  int sx = 0, sy = 0, sz = 0;
  for (int i = 0; i < n; i++) {
    sx += x[i];
    sy += y[i];
    sz += z[i];
  }
  if (!sx && !sy && !sz)
    cout << "YES"
         << "\n";
  else
    cout << "NO"
         << "\n";
  return 0;
}
