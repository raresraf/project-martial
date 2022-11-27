/**
Coder : Iman Mousaei

Accepted
**/

#include <bits/stdc++.h>
#define ___ ios_base::sync_with_stdio(false);

using namespace std;

typedef long long ll;
const ll LLINF = LLONG_MAX - 7;
const int INF = INT_MAX - 7;

const int MAX = 1e9 + 7;
const int MOD = 1e9 + 7;

int main() {
  int w;
  cin >> w;
  if (w % 4 == 0 || ((w - 2) % 2 == 0 && (w - 2) > 0))
    cout << "YES";
  else
    cout << "NO";

  return EXIT_SUCCESS;
}
