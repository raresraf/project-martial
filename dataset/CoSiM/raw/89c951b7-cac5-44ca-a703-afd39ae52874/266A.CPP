#include <bits/stdc++.h>
#define lo                                                                     \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);
using namespace std;

int main() {
  lo int x;
  cin >> x;
  string s;
  cin >> s;
  int res = 0;
  for (int i = 0; i < x - 1; i++) {
    if (s[i] == s[i + 1]) {
      res++;
    }
  }
  cout << res << endl;
  return 0;
}