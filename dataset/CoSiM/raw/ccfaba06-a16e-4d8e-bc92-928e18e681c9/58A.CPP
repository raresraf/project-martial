#include <bits/stdc++.h>
using namespace std;
int main() {
  string str, tmp = "hello";
  int k = 0;
  cin >> str;
  for (int i = 0; i < str.size(); i++) {
    if (k == tmp.size())
      break;
    if (tmp[k] == str[i])
      k++;
  }
  if (k == tmp.size()) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}
