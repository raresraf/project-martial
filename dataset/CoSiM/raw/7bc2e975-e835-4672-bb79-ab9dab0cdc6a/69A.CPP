#include <bits/stdc++.h>
using namespace std;

int main() {

  int n;
  cin >> n;

  int sum1 = 0, sum2 = 0, sum3 = 0, x[n + 10], y[n + 10], z[n + 10];

  for (int i = 0; i < n; i++)
    cin >> x[i] >> y[i] >> z[i];
  for (int i = 0; i < n; i++) {
    sum1 += x[i];
    sum2 += y[i];
    sum3 += z[i];
  }
  if (sum1 == 0 && sum2 == 0 && sum3 == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
