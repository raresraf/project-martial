#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
  int n, t;
  scanf("%d%d", &n, &t);
  string s;
  cin >> s;

  for (int i = 1; i <= t; i++) {
    for (int j = 0; j < n - 1; j++) {
      if (s[j] == 'B' && s[j + 1] == 'G') {
        s[j + 1] = 'B';
        s[j] = 'G';
        j++;
      }
    }
  }
  cout << s << "\n";
  return 0;
}