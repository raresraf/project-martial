#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  string x;
  cin >> x;
  int flg = -1;
  int count = 0;
  for (int i = 0; i < n; i++) {
    if (x.at(i) == 'R') {
      if (flg == 0)
        count++;
      flg = 0;
    } else if (x.at(i) == 'G') {
      if (flg == 1)
        count++;
      flg = 1;
    } else if (x.at(i) == 'B') {
      if (flg == 2)
        count++;
      flg = 2;
    }
  }
  cout << count;
}