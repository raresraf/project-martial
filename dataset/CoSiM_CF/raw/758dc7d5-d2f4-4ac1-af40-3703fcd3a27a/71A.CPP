#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <stdio.h>
#include <string.h>
#include <string>
#include <tuple>
#include <vector>
using namespace std;

#define INF 987654321
#define mod 1000000007
#define mt make_tuple
#define mp make_pair
typedef long long ll;

int main() {

  int n;
  scanf("%d", &n);
  char a[101];
  for (int i = 0; i < n; i++) {
    scanf("%s", a);
    int len = strlen(a);
    if (len > 10) {
      printf("%c%d%c\n", a[0], len - 2, a[len - 1]);
    } else {
      printf("%s\n", a);
    }
  }

  return 0;
}