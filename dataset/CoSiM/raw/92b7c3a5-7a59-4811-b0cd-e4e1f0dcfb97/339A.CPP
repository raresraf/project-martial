#include <bits/stdc++.h>
using namespace std;
int main() {
  int arr[101];
  int cnt = 0;
  string s;
  cin >> s;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == '+') {
      continue;
    } else {
      arr[cnt++] = s[i] - '0';
    }
  }
  sort(arr, arr + cnt);
  for (int i = 0; i < cnt; i++) {
    cout << arr[i];
    if (i == cnt - 1) {
      break;
    }
    cout << "+";
  }
}