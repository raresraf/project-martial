#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, x;
  string s;
  scanf("%d", &n);
  x = 0;
  while (n--) {
    cin >> s;
    if (s[1] == '+')
      x++;
    else
      x--;
  }
  printf("%d", x);
}
