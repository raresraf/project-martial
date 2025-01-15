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

int arr[51];

int main() {

  int n, m;
  scanf("%d %d", &n, &m);
  for (int i = 0; i < n; i++) {
    scanf("%d", &arr[i]);
  }
  sort(arr, arr + n, [](int a, int b) { return a > b; });
  int point = arr[m - 1];
  int ans = 0;
  int i = 0;
  while (point <= arr[i] && arr[i] > 0) {
    ans++, i++;
  }
  printf("%d\n", ans);

  return 0;
}