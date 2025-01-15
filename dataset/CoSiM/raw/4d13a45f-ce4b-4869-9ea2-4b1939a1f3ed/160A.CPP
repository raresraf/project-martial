#include <algorithm>
#include <bits/stdc++.h>
#include <bits/stl_algo.h>
using namespace std;

int main() {
  int coin = 0, count = 0, hasil = 0, i, a[101], n;
  scanf("%d", &n);
  for (i = 1; i <= n; i++) {
    scanf("%d", &a[i]);
    count += a[i];
  }
  sort(a, a + n + 1); // setelah diurutkan i mulai dari 0 a[0]. jika n=4 maka
                      // a[0] sampai a[3]
  count = count / 2;
  for (i = n - 1; i >= 0; i--) {
    if (hasil <= count) {
      coin++;
      hasil = hasil + a[i];
    }
  }
  cout << coin << endl;
}
