#include <bits/stdc++.h>
#define pi 2.0 * acos(0.0)
#define MAX 100005
typedef long long ll;
using namespace std;

int main() {
  int a[2000][3], count[MAX], i, j, k, n, c;
  c = 0;
  cin >> n;
  for (i = 0; i < n; i++)
    count[i] = 0;
  for (i = 0; i < n; i++) {
    for (j = 0; j < 3; j++) {
      cin >> a[i][j];
    }
  }
  for (i = 0; i < n; i++) {
    for (j = 0; j < 3; j++) {
      if (a[i][j] == 1)
        count[i]++;
    }
  }
  for (i = 0; i < n; i++) {
    if (count[i] >= 2) {
      c = c + 1;
    }
  }
  cout << c;
  return 0;
}
