#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main() {
  int n;
  cin >> n;
  int a[n];
  int mi = n - 1;
  int pa = 0;
  int pi = 0;
  int ris = 1;
  for (int i = 0; i < n; i++)
    cin >> a[i];
  sort(a, a + n);
  int ma = a[mi];
  for (int i = 0; i < n; i++) {
    pa += a[i];
    if (pa >= ma) {
      if (mi > i) {
        ris++;
        mi--;
        ma += a[mi];
      } else {
        cout << ris << "\n";
        return 0;
      }
    } else if (i >= mi - 1) {
      cout << ris << "\n";
      return 0;
    }
  }
  return 0;
}
