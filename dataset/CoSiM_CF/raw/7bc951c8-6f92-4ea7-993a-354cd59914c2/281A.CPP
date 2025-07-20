#include <bits/stdc++.h>
using namespace std;
int main() {
  int cnt = 0;
  int arr[1001];
  string s;
  cin >> s;
  if (s[0] >= 97 && s[0] <= 122) {
    s[0] -= 32;
  }
  cout << s;
}