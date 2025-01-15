#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, t, i, j;
  char a[101];
  scanf("%d %d", &n, &t);
  for (i = 1; i <= n; i++) {
    scanf("%s", &a[i]);
  }
  for (j = 1; j <= t; j++) {

    for (i = n; i >= 1;) {
      if (a[i] == 'B' && i != n) {
        swap(a[i], a[i + 1]);
        i = i - 2;
      } else {
        i--;
      }
    }
  }
  for (i = 1; i <= n; i++) {
    cout << a[i];
  }
  printf("\n");
  return 0;
}
