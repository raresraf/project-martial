// under 3 min solved
#include <bits/stdc++.h>
using namespace std;

int main() {
  int c = 0;
  int m, n;
  cin >> m >> n;
  int r = m / 2;
  c = r * n;
  if (m % 2 != 0)
    c += (n / 2);
  cout << c << endl;
}
