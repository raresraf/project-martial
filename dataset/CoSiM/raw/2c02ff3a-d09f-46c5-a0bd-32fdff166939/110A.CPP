#include <bits/stdc++.h>
using namespace std;
int main() {
  string str;
  int n, cnt = 0;
  cin >> str;
  n = str.size();
  for (int i = 0; i < n; i++) {
    if (str[i] == '7' || str[i] == '4') {
      cnt++;
    }
  }
  if (cnt == 7 || cnt == 4) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
