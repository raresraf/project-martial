#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  int n;
  cin >> n;
  while (n--) {
    cin >> s;
    int l = s.size();
    if (l > 10)
      printf("%c%d%c\n", s[0], l - 2, s[l - 1]);
    else
      cout << s << endl;
  }
  return 0;
}
