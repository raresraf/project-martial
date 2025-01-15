#include <bits/stdc++.h>

using namespace std;

int main() {
  long long i;
  cin >> i;
  int numLucky = 0;
  while (i != 0) {
    if (i % 10 == 4 || i % 10 == 7)
      numLucky++;
    i = i / 10;
  }
  if (numLucky == 4 || numLucky == 7)
    printf("YES");
  else
    printf("NO");
}