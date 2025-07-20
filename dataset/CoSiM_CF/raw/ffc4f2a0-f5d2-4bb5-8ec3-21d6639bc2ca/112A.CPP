#include <bits/stdc++.h>
#include <iostream>
#include <string.h>
using namespace std;
int main() {
  string a, b;

  cin >> a >> b;
  transform(a.begin(), a.end(), a.begin(), ::tolower);
  transform(b.begin(), b.end(), b.begin(), ::tolower);
  // cout << a << endl << b;
  if (a.compare(b) == -1)
    cout << -1 << endl;
  if (a.compare(b) == 0)
    cout << 0 << endl;
  if (a.compare(b) == 1)
    cout << 1 << endl;

  return 0;
}
