#include <bits/stdc++.h>

using namespace std;

int main() {
  int count1 = 1;
  string s;
  getline(cin, s);
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == s[i + 1]) {
      count1++;
      if (count1 >= 7) {
        break;
      }
    } else {
      s[i] = s[i + 1];
      count1 = 1;
    }
  }
  if (count1 > 6) {
    printf("YES\n");
  } else {
    printf("NO\n");
  }

  return 0;
}
