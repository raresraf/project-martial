#include <bits/stdc++.h>

using namespace std;

int main() {
  long int n, mn, mx;
  cin >> n;
  mn = (n / 7) * 2;
  mx = mn;
  if (n % 7 == 1)
    mx = mn + 1;
  else if (n % 7 > 1)
    mx = mn + 2;
  if (n % 7 == 6)
    mn = mn + 1;
  cout << mn << " " << mx << "\n";
}
