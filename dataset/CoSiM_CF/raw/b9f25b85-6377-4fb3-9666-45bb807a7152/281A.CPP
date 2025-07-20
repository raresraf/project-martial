#include <bits/stdc++.h>

using namespace std;

int main() {
  string a;
  cin >> a;
  if (!('A' <= a[0] && a[0] <= 'Z'))
    a[0] += 'A' - 'a';
  cout << a << endl;
  return 0;
}
