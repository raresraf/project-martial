#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main() {
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  string s1 = s;
  for (int j = 0; j < t; j++) {
    bool flag = false;
    for (int i = n - 1; i >= 0; i--) {
      if (s[i] == 'B' && flag) {
        s1[i + 1] = 'B';
        s1[i] = 'G';
        flag = false;
      } else if (s[i] == 'G')
        flag = true;
    }
    s = s1;
  }
  cout << s << "\n";
  return 0;
}
