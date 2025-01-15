#include <bits/stdc++.h>

using namespace std;
int main() {
  int n, a, b, total = 0, m = 0;
  cin >> n;
  while (n--) {
    cin >> a >> b;
    total -= a;
    total += b;
    if (total > m) {
      m = total;
    }
  }

  cout << m << endl;
}
