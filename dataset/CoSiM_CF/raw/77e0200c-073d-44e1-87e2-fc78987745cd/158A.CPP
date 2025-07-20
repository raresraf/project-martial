
/*                     */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int t, a[101], n, k, f = 0;
  cin >> t >> k;

  for (int i = 0; i < t; i++)
    cin >> a[i];

  for (int i = 0; i < t; i++) {
    if (a[i] >= a[k - 1] && a[i] != 0)
      f++;
  }
  cout << f << endl;

  return 0;
}
