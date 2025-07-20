#include <bits/stdc++.h>
#define _                                                                      \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);

using namespace std;

string toLowercase(string a) {
  string ans = a;
  for (int i = 0; i < a.length(); i++) {
    if (a[i] <= 'Z' && a[i] >= 'A') {
      ans[i] = a[i] - 'A' + 'a';
    }
  }
  return ans;
}

int main() {
  _ string a, b;
  cin >> a >> b;
  a = toLowercase(a);
  b = toLowercase(b);
  if (a == b) {
    cout << "0\n";
  } else if (a < b) {
    cout << "-1\n";
  } else {
    cout << "1\n";
  }
  return 0;
}