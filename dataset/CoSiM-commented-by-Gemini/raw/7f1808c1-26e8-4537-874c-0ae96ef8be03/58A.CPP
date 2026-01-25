#include <bits/stdc++.h>
using namespace std;

int main() {
  string s, r = "hello";
  int i, j = 0;
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == r[j])
      j++;
  }
  if (j == 5)
    printf("YES");
  else
    printf("NO");
}
