#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  int t, i, j, k;
  int a[1001][2], a1[1001];
  cin >> t;
  for (i = 0; i < t; i++) {
    for (j = 0; j < 2; j++) {
      cin >> a[i][j];
    }
  }
  k = 0;
  for (i = 0; i < t; i++) {
    k -= a[i][0];
    k += a[i][1];
    a1[i] = k;
  }
  sort(a1, a1 + t);
  cout << a1[t - 1] << endl;
  return 0;
}
