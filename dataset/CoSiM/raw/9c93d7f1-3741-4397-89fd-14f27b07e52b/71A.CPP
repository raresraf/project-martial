#include <bits/stdc++.h>
using namespace std;
int main() {
  // freopen("input.txt", "r", stdin);
  int t;
  string s;
  for (scanf("%d", &t); t--;) {
    cin >> s;
    if (s.size() <= 10)
      cout << s << endl;
    else
      cout << s[0] << (s.size() - 2) << s[s.size() - 1] << endl;
  }
  return 0;
}
