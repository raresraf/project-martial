#include <bits/stdc++.h>
#include <string>

using namespace std;
int main() {
  string str;
  cin >> setw(101) >> str;
  int flag = 0, one(0), zero(0);
  for (int i = 0; i < str.size(); i++) {

    if (str[i] == '1') {
      one++;
      if (zero != 0) {
        zero = 0;
      }
    } else if (str[i] == '0') {
      zero++;
      if (one != 0) {
        one = 0;
      }
    }
    if (one == 7 || zero == 7) {
      cout << "YES" << endl;
      flag = 1;
      break;
    }
  }
  if (flag == 0) {
    cout << "NO" << endl;
  }
}
