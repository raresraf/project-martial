#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, kq = 0;
  cin >> n;
  while (n--) {
    int dem = 0;
    int a[3];
    for (int i = 0; i < 3; i++) {
      cin >> a[i];
      if (a[i] == 1)
        dem++;
    }
    if (dem >= 2)
      kq++;
  }
  cout << kq;
  return 0;
}
