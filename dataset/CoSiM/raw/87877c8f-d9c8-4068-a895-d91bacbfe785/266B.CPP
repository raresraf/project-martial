#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, t;
  cin >> n >> t;
  string s;
  while (n--) {
    char aux;
    cin >> aux;
    s.push_back(aux);
  }
  while (t--) {
    for (int i = 0; i < s.size() - 1; i++) {
      if (s[i] == 'B' and s[i + 1] == 'G') {
        s[i] = 'G';
        s[i + 1] = 'B';
        i = i + 1;
      }
    }
  }
  cout << s << endl;
  return 0;
}
