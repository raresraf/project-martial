#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main() {
  int n;
  cin >> n;
  int c = 0;
  int m = 0;
  int a, b;
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    c -= a;
    c += b;
    if (c > m)
      m = c;
  }
  cout << m << "\n";
  return 0;
}
