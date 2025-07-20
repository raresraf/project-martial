#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, t;
  char s[55];
  scanf("%d%d", &n, &t);
  scanf("%s", s);
  for (int i = 0; i < t; i++)
    for (int j = 1; j < n; j++)
      if (s[j - 1] == 'B' && s[j] == 'G')
        swap(s[j - 1], s[j]), j++;
  printf("%s", s);
  return 0;
}