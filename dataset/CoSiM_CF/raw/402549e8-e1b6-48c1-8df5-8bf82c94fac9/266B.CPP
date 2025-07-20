#include <bits/stdc++.h>
using namespace std;
typedef long int l;
int main() {
  int n, t, k;
  cin >> n >> t;
  string a;
  cin >> a;
  for (int i = 0; i < t; i++) {
    k = 0;
    if (k == n)
      break;
    for (int j = 0; j < n - 1;) {
      k++;
      if (a[j] == 'B' && a[j + 1] == 'G') {
        a[j] = 'G';
        a[j + 1] = 'B';
        j += 2;
      } else
        j++;
    }
  }
  cout << a;
}
