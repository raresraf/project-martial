#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <utility>
#include <vector>

using namespace std;

typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<pii> vii;
#define mod 1000000007
#define s(a) scanf("%d", &a)
#define sll(a) scanf("%lld", &a)
#define ss(a) scanf("%s", a)
#define ll long long
#define mp make_pair
#define pb push_back
#define fi first
#define se second

char a[55];

int main() {
  int n, t;

  s(n), s(t);
  scanf("%s", a);

  for (int i = 0; i < t; ++i) {
    for (int j = n - 1; j > 0; --j) {
      if (a[j] == 'G' && a[j - 1] == 'B') {
        swap(a[j], a[j - 1]);
        j--;
      }
    }
  }

  printf("%s\n", a);

  return 0;
}