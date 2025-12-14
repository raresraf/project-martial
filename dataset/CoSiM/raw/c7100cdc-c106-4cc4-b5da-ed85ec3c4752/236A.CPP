#include <bits/stdc++.h>
#include <ctype.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  int len = s.size();
  int a[26] = {0}, count = 0;
  for (int i = 0; i < len; i++) {
    int x = (s[i] - 'a');
    if (a[x] == 0) {
      a[x]++;
      count++;
    }
  }
  if (count % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
  return 0;
}
