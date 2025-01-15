#include <bits/stdc++.h>
using namespace std;

void dihan() {
  long long a, b, c = 0;
  cin >> a >> b;

  c = (a + b) / 3;

  if (c <= min(a, b))
    cout << c << endl;

  else
    cout << min(a, b) << endl;
}

int main() {
  int t;
  cin >> t;
  while (t--)
    dihan();
}
