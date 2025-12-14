#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, ans = 0;
  string inp;
  scanf("%d", &n);
  getline(cin, inp);
  cin >> inp;
  for (int i = 1; i < n; i++)
    if (inp[i] == inp[i - 1])
      ans++;
  printf("%d", ans);
  return 0;
}