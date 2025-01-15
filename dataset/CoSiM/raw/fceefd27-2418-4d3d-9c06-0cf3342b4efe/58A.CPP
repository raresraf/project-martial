#include <bits/stdc++.h>
using namespace std;
int main() {
  string str;
  int cnt = 0, l;
  cin >> str;
  l = str.size();
  for (int i = 0; i < l; i++) {
    if (cnt == 0 && str[i] == 'h') {
      cnt++;
      // cout<<cnt<<str[i]<<endl;
    } else if (cnt == 1 && str[i] == 'e') {
      cnt++;
      // cout<<cnt<<str[i]<<endl;
    } else if (cnt >= 2 && cnt <= 3 && str[i] == 'l') {
      cnt++;
      // cout<<cnt<<str[i]<<endl;
    } else if (cnt == 4 && str[i] == 'o') {
      cnt++;
      // cout<<cnt<<str[i]<<endl;
    } else {
    }
  }
  if (cnt == 5) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
