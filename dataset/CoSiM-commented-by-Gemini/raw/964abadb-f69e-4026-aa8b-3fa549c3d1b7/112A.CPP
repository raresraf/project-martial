#include <bits/stdc++.h>
using namespace std;
int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);

  string s, s1;
  cin >> s >> s1;

  transform(s.begin(), s.end(), s.begin(), ::tolower);
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  if (s > s1)
    cout << "1\n";
  if (s == s1)
    cout << "0\n";
  if (s < s1)
    cout << "-1\n";
}
