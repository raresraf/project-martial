#include <bits/stdc++.h>
using namespace std;

int main() {
  string str;
  int n, t;

  cin >> n >> t;
  cin >> str;

  while (t--) {
    for (int i = 1; i < n; i++) {
      if (str[i] == 'G' && str[i - 1] == 'B') {
        str[i] = 'B';
        str[i - 1] = 'G';
        i++;
      }
    }
  }
  cout << str << endl;
  return 0;
}
