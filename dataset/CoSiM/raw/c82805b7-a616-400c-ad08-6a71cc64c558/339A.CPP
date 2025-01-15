// CF https://codeforces.com/problemset/problem/339/A

#include <bits/stdc++.h>

using namespace std;

int32_t main() {

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  string s;
  getline(cin, s);
  int n = s.length();
  int arr[n];
  int j = 0;

  for (char x : s) {
    if (x != '+') {
      arr[j++] = abs(48 - int(x));
    }
  }
  sort(arr, arr + j);
  for (int i = 0; i < j; i++) {
    cout << arr[i];
    if (i != j - 1) {
      cout << "+";
    }
  }
  return 0;
}
