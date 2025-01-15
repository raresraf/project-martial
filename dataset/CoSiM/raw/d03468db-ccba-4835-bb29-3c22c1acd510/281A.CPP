// https://codeforces.com/problemset/problem/281/A
// SOLUTION BY DN54321

#include <ctype.h>
#include <iostream>
#include <string.h>
#include <vector>
typedef long long ll;
using namespace std;
int main() {
  char s[10001];
  scanf("%s", &s);
  printf("%c%s", toupper(s[0]), &s[1]);
  return 0;
}