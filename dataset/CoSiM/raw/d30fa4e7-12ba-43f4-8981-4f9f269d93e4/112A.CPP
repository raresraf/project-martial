#include <bits/stdc++.h>
#include <cctype>
#include <string>
#define ll long long
using namespace std;
int main() {

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  int n, z = 0;
  string a, b;
  cin >> a >> b;
  transform(a.begin(), a.end(), a.begin(), ::tolower);
  transform(b.begin(), b.end(), b.begin(), ::tolower);
  for (int i = 0; i < a.length(); i++) {
    if (int(a[i]) == int(b[i])) {
      z = 0;
    } else if (int(a[i]) > int(b[i])) {
      z = 1;
      break;
    } else if (int(a[i]) < int(b[i])) {
      z = -1;
      break;
    }
  }
  cout << z;
  return 0;
}
