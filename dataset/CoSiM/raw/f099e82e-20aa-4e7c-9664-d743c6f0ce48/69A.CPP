/***

            Bismillahir Rahmanir Rahim
            Read the name of Allah, who created you!!!
            Author : Shah Newaj Rabbi Shishir
            Department of CSE, City University, Bangladesh.

***/

#include <bits/stdc++.h>
using namespace std;

#define sf scanf
#define pf printf
#define scase sf("%d", &tc)
#define sn sf("%d", &n)
#define whilecase while (tc--)
#define eof while (cin >> n)
#define forloop for (pos = 1; pos <= tc; pos++)
#define arrayloop (i = 0; i < n; i++)
#define cinstr cin >> str
#define getstr getline(cin, str)
#define pcase pf("Case %d: ", pos)
#define pb push_back
#define in insert
#define llu unsigned long long
#define lld long long
#define u unsigned int

int main(void) {
  int n, x, y, z, X, Y, Z;

  while (cin >> n) {
    X = Y = Z = 0;

    while (n--) {
      cin >> x >> y >> z;

      X += x;
      Y += y;
      Z += z;
    }

    if (X == 0 && Y == 0 && Z == 0)
      cout << "YES\n";
    else
      cout << "NO\n";
  }

  return 0;
}
