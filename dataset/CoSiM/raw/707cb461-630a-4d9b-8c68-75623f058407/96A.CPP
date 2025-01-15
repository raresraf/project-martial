#include <bits/stdc++.h>
using namespace std;
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  string str;
  long long countt = 0, counti = 0;
  getline(cin, str);
  int n = str.length();
  char charr[n + 1];
  strcpy(charr, str.c_str());
  for (int i = 0; i < n + 1; i++) {
    if (charr[i] == '0') {
      countt = 0;
      counti = counti + 1;
      if (counti == 7) {
        break;
      }
    } else if (charr[i] == '1') {
      counti = 0;
      countt = countt + 1;
      if (countt == 7) {

        break;
      }
    }
  }
  if (counti == 7 || countt == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
