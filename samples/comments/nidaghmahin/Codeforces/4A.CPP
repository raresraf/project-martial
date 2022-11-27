#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <string>
#include <vector>
using namespace std;
#define READ(f)
#define WRITE(f)
#define ALL(c) c.begin(), c.end()
#define PB(x) push_back(x)
#define i64 long long
#define ld long double

int main() {
  // freopen("in.txt", "r", stdin);
  // freopen("out.txt", "w", stdout);

  int n;
  scanf("%d", &n);
  if (n <= 2 || n % 2 == 1)
    cout << "NO\n";
  else
    cout << "YES\n";
}