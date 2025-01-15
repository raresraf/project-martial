#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int f = 0;
void dp(string s, string h, int i, int j) {
  if (i >= s.size() || j >= h.size())
    return;
  if (s[i] == h[j]) {
    dp(s, h, i + 1, j + 1);
    if (j + 1 == h.size())
      f = 1;
  } else
    dp(s, h, i + 1, j);
}
int main() {
  string s, h = "hello";
  cin >> s;
  dp(s, h, 0, 0);
  if (f == 1)
    cout << "YES";
  else
    cout << "NO";
}
