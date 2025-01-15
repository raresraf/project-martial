#include <bits/stdc++.h>
using namespace std;
int main() {
  string x;
  cin >> x;
  int n0 = 0, n1 = 0;
  int flg = 0;
  for (int i = 0; i < x.length(); ++i) {
    if (x.at(i) == '0') {
      n0++;
      n1 = 0;
    }
    if (x.at(i) == '1') {
      n1++;
      n0 = 0;
    }
    if (n0 == 7 || n1 == 7) {
      flg = 1;
      break;
    }
  }
  if (flg == 1)
    cout << "YES";
  else
    cout << "NO";
}