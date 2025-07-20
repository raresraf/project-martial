#include <bits/stdc++.h>
using namespace std;
int main() {
  int m = 0, n = 0;
  string a, b;
  cin >> a >> b;
  transform(a.begin(), a.end(), a.begin(), ::toupper);
  transform(b.begin(), b.end(), b.begin(), ::toupper);

  if (a < b)
    cout << -1 << endl;
  else if (a > b)
    cout << 1 << endl;
  else
    cout << 0 << endl;
}