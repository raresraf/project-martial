#include <bits/stdc++.h>
using namespace std;

int in[1000];

int main() {
  string s;
  cin >> s;
  int len = s.length();
  int j;
  for (j = 0; j < len; j++) {
    if (s[j] != '+')
      in[j] = s[j] - 48;
  }
  sort(in, in + len);
  int j1;
  for (j1 = 0; j1 < len; j1++) {
    if (in[j1] != 0) {
      printf("%d", in[j1]);
      if (j1 == len - 1)
        printf("\n");
      else
        printf("+");
    }
  }
  return 0;
}
