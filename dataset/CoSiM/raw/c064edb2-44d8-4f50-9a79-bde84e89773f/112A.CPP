#include <bits/stdc++.h>
using namespace std;

int main() {

  string s1, s2;
  int c1 = 0, c2 = 0;
  cin >> s1 >> s2;
  for (char &i : s1) {
    if (i < 'a')
      i += 32;
  }
  for (char &i : s2) {
    if (i < 'a')
      i += 32;
  }
  cout << (s1.compare(s2)) << endl;

  return 0;
}
