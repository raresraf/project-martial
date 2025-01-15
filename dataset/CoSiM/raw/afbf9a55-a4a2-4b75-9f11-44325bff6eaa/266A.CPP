#include <bits/stdc++.h>
using namespace std;

int main(void) {

  // freopen("c.in" , "r" , stdin);
  int t = 0;

  cin >> t;
  string s;
  cin >> s;
  int counter = 0;
  for (int i = 0, j = s.length(); i < j - 1; i++) {
    if (s[i] == s[i + 1]) {
      counter++;
    }
  }
  cout << counter << endl;
  return 0;
}
