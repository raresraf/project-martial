#include <bits/stdc++.h>
using namespace std;

int main() {
  char a[100];
  cin >> a;
  int n = strlen(a);
  int x = 0, y = 0;
  for (int i = 0; i < n; i++) {
    int c = a[i];
    c -= 48;
    if (c == 1) {
      x++;
      y = 0;
    } else {
      y++;
      x = 0;
    }
    if (x == 7 || y == 7)
      break;
  }
  if (x == 7 || y == 7) {
    cout << "YES" << endl;
  } else
    cout << "NO" << endl;

  return 0;
}
