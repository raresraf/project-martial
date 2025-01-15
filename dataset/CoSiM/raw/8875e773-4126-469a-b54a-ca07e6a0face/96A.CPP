#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  int pos, pos1;
  while (cin >> s) {
    pos = s.find("0000000");
    pos1 = s.find("1111111");

    if (pos != -1 || pos1 != -1)
      cout << "YES" << endl;
    else
      cout << "NO" << endl;
  }

  return 0;
}
