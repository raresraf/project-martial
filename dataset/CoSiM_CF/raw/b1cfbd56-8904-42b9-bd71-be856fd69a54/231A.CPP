#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, p, v, t, counter = 0;
  cin >> n;

  for (int i = 0; i < n; ++i) {
    cin >> p >> v >> t;

    if (p + v + t >= 2) {
      counter++;
    }
  }

  cout << counter;
  return 0;
}
