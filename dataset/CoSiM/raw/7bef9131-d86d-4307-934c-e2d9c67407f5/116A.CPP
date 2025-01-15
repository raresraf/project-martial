#include <bits/stdc++.h>
using namespace std;

int max(int a, int b) { return a > b ? a : b; }

int main() {
  int n;
  cin >> n;

  int tpax = 0;
  int pax = 0;

  for (int i = 0; i < n; i++) {
    int a, b;
    cin >> a >> b;

    pax -= a;
    pax += b;

    tpax = max(tpax, pax);
  }

  cout << tpax << endl;

  return 0;
}
