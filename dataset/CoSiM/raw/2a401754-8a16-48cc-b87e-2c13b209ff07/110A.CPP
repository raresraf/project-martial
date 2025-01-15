#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);

  long long n;
  cin >> n;

  int c(0), c1(0);

  stringstream ss;
  ss << n;
  string s;
  ss >> s;

  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '4' || s[i] == '7')
      c++;
  }

  if (c == 4 || c == 7)
    cout << "YES\n";

  else
    cout << "NO\n";
}
