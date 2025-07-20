#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, t;
  char s[55];
  scanf("%d%d", &n, &t);
  scanf("%s", s);

  for (int i = 0; i < t; i++) {
    for (int j = 0; j < n - 1; j++) {
      if (s[j] == 'B' && s[j + 1] == 'G') {
        swap(s[j], s[j + 1]);
        j++;
      }
    }
  }
  printf("%s", s);
  return 0;
}
