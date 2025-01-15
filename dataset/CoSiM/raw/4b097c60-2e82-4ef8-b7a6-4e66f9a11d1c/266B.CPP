#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  for (int i = 0; i < t; i++)
    for (int j = 0; j < n; j++) {
      if (s[j] == 'B')
        if (s[j + 1] == 'G') {
          char c = s[j];
          s[j] = s[j + 1];
          s[j + 1] = c;
          j++;
        }
    }
  cout << s;
  return 0;
}