#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, tram = 0, a, b, max = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    tram = tram - a + b;
    if (tram > max) {
      max = tram;
    }
  }
  cout << max;

  return 0;
}
