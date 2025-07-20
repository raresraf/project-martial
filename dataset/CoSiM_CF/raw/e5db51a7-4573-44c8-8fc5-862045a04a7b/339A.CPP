#include <bits/stdc++.h>
using namespace std;
int main() {
  string x;
  cin >> x;
  int n1 = 0, n2 = 0, n3 = 0;
  for (int i = 0; i < x.length(); i++) {
    if (x.at(i) == '1')
      n1++;
    else if (x.at(i) == '2')
      n2++;
    else if (x.at(i) == '3')
      n3++;
  }
  string ans = "";
  while (1) {
    if (n1 == 0 && n2 == 0 && n3 == 0)
      break;
    if (ans.length() == 0) {
      if (n1 > 0) {
        ans = ans + '1';
        n1--;
      } else if (n2 > 0) {
        ans = ans + '2';
        n2--;
      } else if (n3 > 0) {
        ans = ans + '3';
        n3--;
      }
    } else {
      if (n1 > 0) {
        ans = ans + "+1";
        n1--;
      } else if (n2 > 0) {
        ans = ans + "+2";
        n2--;
      } else if (n3 > 0) {
        ans = ans + "+3";
        n3--;
      }
    }
  }
  cout << ans;
}