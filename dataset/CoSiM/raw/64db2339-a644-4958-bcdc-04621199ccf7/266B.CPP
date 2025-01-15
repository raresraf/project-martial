#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, k;
  cin >> n >> k;
  string s;
  cin >> s;
  while (k--) {
    string z = s;
    for (int i = 1; i < s.length(); i++)
      if (s[i - 1] == 'B' && s[i] == 'G')
        swap(z[i - 1], z[i]);
    s = z;
  }
  cout << s << endl;
  return 0;
}
