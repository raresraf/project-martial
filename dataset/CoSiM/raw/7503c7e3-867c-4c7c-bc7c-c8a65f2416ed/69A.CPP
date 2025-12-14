#include <bits/stdc++.h>
using namespace std;

int main() {
  // freopen("file.text", "r", stdin);
  int n;
  cin >> n;
  int x[n + 1], y[n + 1], z[n + 1];

  for (int i = 0; i < n; i++)
    cin >> x[i] >> y[i] >> z[i];

  int sumX = 0, sumY = 0, sumZ = 0;
  for (int i = 0; i < n; i++) {
    sumX += x[i];
    sumY += y[i];
    sumZ += z[i];
  }

  if (sumX == 0 && sumY == 0 && sumZ == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
