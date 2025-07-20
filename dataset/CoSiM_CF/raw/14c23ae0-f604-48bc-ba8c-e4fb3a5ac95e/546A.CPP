#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);

  int a, b, c, sum(0);
  cin >> a >> b >> c;

  for (int i = 1; i <= c; i++) {
    sum += (a * i);
  }
  if (sum > b)
    cout << sum - b << "\n";

  else
    cout << "0\n";
}
