#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  int luckyCount = 0;
  cin >> s;
  for (char a : s) {
    if (a == '4' || a == '7') {
      luckyCount++;
    }
  }
  (luckyCount == 4 || luckyCount == 7) ? cout << "YES" : cout << "NO";
}
// notaslucky
