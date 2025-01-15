#include <bits/stdc++.h>
using namespace std;
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  string str;
  long long counti = 0, n;
  cin >> n;
  cin >> str;
  char charr[n + 1];
  strcpy(charr, str.c_str());
  for (int i = 0; i < n; i++) {
    if (charr[i] == charr[i + 1]) {
      counti = counti + 1;
    }
  }
  cout << counti;
  return 0;
}
