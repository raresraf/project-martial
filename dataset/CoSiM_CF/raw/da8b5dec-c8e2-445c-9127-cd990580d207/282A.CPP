/*
 * author: rakib_
 */

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef long double ld;
#define fast_io                                                                \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);

int main() {
  fast_io;
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  int n;
  cin >> n;

  int x = 0;

  while (n--) {
    string str;
    cin >> str;

    if (str[0] == '+' || str[2] == '+')
      x++;
    else
      x--;
  }

  cout << x << endl;

  return 0;
}