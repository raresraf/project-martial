#include <bits/stdc++.h>
using namespace std;
int main() {
  int count, n;
  scanf("%d", &n);
  char stone[n + 1];
  scanf("%s", &stone);
  count = 0;
  for (int i = 1; i < strlen(stone); i++) {
    if (stone[i] == stone[i - 1])
      count++;
  }
  printf("%d", count);
}