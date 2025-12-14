#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, pas = 0, maks = 0;
  cin >> n;
  while (n--) {
    int a, b;
    cin >> a >> b;
    pas -= a;
    pas += b;
    if (pas > maks)
      maks = pas;
  }
  cout << maks << "\n";
}
