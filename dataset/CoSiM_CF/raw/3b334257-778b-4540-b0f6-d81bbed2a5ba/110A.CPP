#include <bits/stdc++.h>
using namespace std;
int main() {
  char s[100];
  int i, cnt = 0, len;
  scanf("%s", &s);
  len = strlen(s);
  for (i = 0; i < len; i++) {
    if (s[i] == '4' || s[i] == '7')
      cnt++;
  }
  if (cnt == 4 || cnt == 7) {
    cout << "YES" << endl;
  } else
    cout << "NO" << endl;

  return 0;
}
