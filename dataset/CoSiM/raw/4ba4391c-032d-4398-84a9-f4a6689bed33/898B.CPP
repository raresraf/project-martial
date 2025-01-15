#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int n, a, b, n1, n2, n3, a1 = 0, a2 = 0, i;
  cin >> n >> a >> b;
  if (a < b) {
    n1 = a;
    n2 = b;
  } else {
    n1 = b;
    n2 = a;
  }
  n3 = n / n1;
  for (i = 0; i <= n3; i++) {
    if ((n - n1 * i) % n2 == 0) {
      if (a == n1) {
        a1 = i;
        a2 = (n - n1 * i) / n2;
        break;
      } else {
        a2 = i;
        a1 = (n - n1 * i) / n2;
        break;
      }
    }
  }
  if (a1 == 0 && a2 == 0)
    cout << "NO\n";
  else {
    cout << "YES\n";
    cout << a1 << " " << a2 << endl;
  }
  return 0;
}