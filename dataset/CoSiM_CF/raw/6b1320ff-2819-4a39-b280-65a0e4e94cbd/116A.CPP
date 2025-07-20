#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, i, a[1002], b[1002], c;
  long int max;
  scanf("%d", &n);
  for (i = 1; i <= n; i++) {
    scanf("%d %d", &a[i], &b[i]);
  }
  c = a[1] + b[1];
  max = c;
  for (i = 1; i <= n - 1; i++) {
    c = c - (a[i + 1] - b[i + 1]);
    if (c > max) {
      max = c;
    }
  }
  cout << max << endl;
}
