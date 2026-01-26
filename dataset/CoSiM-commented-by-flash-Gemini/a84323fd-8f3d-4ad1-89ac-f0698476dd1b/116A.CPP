#include <algorithm>
#include <cmath>
#include <iostream>
using namespace std;
int main() {
  int n, m = 0, z = 0;
  cin >> n;
  int a[1000][2] = {0};
  for (int i = 0; i < n; i++)
    cin >> a[i][0] >> a[i][1];
  for (int i = 0; i < n; i++) {
    z += a[i][1] - a[i + 1][0];
    a[i + 1][1] += z;
    m = max(z, a[i][1]);
  }
  cout << n;
}