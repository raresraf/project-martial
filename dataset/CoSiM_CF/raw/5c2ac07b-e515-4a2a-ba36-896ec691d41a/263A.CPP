// https://codeforces.com/problemset/problem/263/A
// SOLUTION BY DN54321

#include <ctype.h>
#include <iostream>
#include <string.h>
#include <vector>
typedef long long ll;
using namespace std;
int main() {
  int v;
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j) {
      scanf("%d", &v);
      if (v == 1)
        printf("%d", abs(i - 2) + abs(j - 2));
    }
  return 0;
}