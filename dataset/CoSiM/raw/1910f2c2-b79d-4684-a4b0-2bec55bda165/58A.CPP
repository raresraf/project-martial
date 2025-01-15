#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;
  char c[] = {'h', 'e', 'l', 'l', 'o'};
  int co = 0;
  for (int u = 0; u < s.size(); u++) {
    if (s[u] == c[co]) {
      co++;
      if (co == 5) {
        cout << "YES";
        return 0;
      }
    }
  }
  cout << "NO";
  return 0;
}
