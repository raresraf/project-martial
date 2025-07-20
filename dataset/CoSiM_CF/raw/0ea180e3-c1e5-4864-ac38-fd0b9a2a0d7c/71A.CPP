#include <bits/stdc++.h>
#include <iostream>
using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(nullptr);

  int n;
  cin >> n;
  while (n--) {
    string s;
    cin >> s;
    if (s.size() > 10)
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << endl;
    else
      cout << s << endl;
  }

  return 0;
}