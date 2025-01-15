#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int n;
  cin >> n;
  long long int a[n][3];
  for (long long int i = 0; i < n; i++) {
    for (long long int j = 0; j < 3; j++)
      cin >> a[i][j];
  }
  long long int sum_x = 0, sum_y = 0, sum_z = 0;
  for (int i = 0; i < n; i++) {
    sum_x += a[i][0];
    sum_y += a[i][1];
    sum_z += a[i][2];
  }
  if (sum_x == 0 && sum_y == 0 && sum_z == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
