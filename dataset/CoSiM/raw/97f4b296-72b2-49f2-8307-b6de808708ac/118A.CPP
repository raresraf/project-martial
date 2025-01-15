#include <bits/stdc++.h>
#define ll long long
#define ull unsigned long long
#define pb push_back

using namespace std;

int main() {
  std::ios_base::sync_with_stdio(false);
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  string s;
  cin >> s;
  string s1(".");
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == 'u' || s[i] == 'o' || s[i] == 'i' || s[i] == 'e' ||
        s[i] == 'a' || s[i] == 'U' || s[i] == 'O' || s[i] == 'I' ||
        s[i] == 'E' || s[i] == 'A') {
      continue;
    } else {
      s1 += s[i];
      s1 += ".";
    }
  }
  for (int i = 0; i < s1.size(); i++) {
    s1[i] = tolower(s1[i]);
  }
  s1.pop_back();
  cout << s1 << "\n";
  return 0;
}
