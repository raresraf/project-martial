#include <bits/stdc++.h>
using namespace std;

int main() {

  int n, a, b, max = 0, tot = 0;
  cin >> n;

  while (n--) {
    cin >> a >> b;
    tot += -a + b;
    if (max < tot)
      max = tot;
  }
  cout << max << endl;
  return 0;
}
