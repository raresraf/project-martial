#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);

  string s, s1;
  cin >> s;

  s1 = s.at(0);
  transform(s1.begin(), s1.end(), s1.begin(), ::toupper);
  s.erase(0, 1);

  cout << s1 + s << "\n";
}
