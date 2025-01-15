/// 45335266	2018-11-05	@Author: Sadikul Islam Shuvo	A - Chat room
/// GNU C++17	Accepted	30 ms	200 KB

#include <bits/stdc++.h>
using namespace std;
int main() {
  int i, j, c;
  string s1, s2;
  s1 = "hello";
  while (cin >> s2) {
    j = 0;
    c = 0;
    for (i = 0; i < s2.size(); i++) {
      if (s1[j] == s2[i]) {
        j++;
        c++;
      }
    }
    if (c == 5) {
      cout << "YES" << endl;
    } else {
      cout << "NO" << endl;
    }
  }

  return 0;
}
