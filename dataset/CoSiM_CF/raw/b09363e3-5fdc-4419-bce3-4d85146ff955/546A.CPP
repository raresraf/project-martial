/*
Link: https://codeforces.com/problemset/problem/546/A
@uthor: chavodon
*/

#include <iostream>

using namespace std;

int main() {
  int n, k, w, tc, totBan = 0, i;
  cin >> k >> n >> w;
  for (i = 1; i <= w; i++)
    totBan += i;
  tc = totBan * k;
  if (tc <= n)
    cout << '0';
  else
    cout << tc - n;
}
