#include <bits/stdc++.h>
using namespace std;
void swap(char *x, char *y);
int i = 0;
void swap(char *x, char *y) {
  char t;
  t = *x;
  *x = *y;
  *y = t;
}
int main() {
  int i, t, n;
  char a[9999];
  cin >> n >> t;
  for (int j = 0; j < n; j++) {
    cin >> a[j];
  }
  for (int l = t; l >= 1; l--) {
    for (i = 0; i < n; i++) {
      if ((a[i] == 'B') && (a[i + 1] == 'G')) {
        swap(a[i], a[i + 1]);
        i++;
      }
    }
  }
  for (int k = 0; k < n; k++) {
    cout << a[k];
  }
  return 0;
}
