/*
        "Karmanye vadhika raste, Ma phaleshu kadachana
        Ma karma phala he tur bhuh, ma te sangotsva karmanye"

        English Meaning ----> You have a right to Karma (actions) but never to
   any Fruits thereof. You should never be motivated by the results of your
   actions, nor should there be any attachment in not doing your prescribed
   activities.

*/
//#include <iostream> if above is slow
#include <bits/stdc++.h>
using namespace std;

int main() {
  int z;
  int x;
  int y;
  cin >> x >> y >> z;
  int ans = ((x * (z + 1) * z / 2 - y) > 0) ? (x * (z + 1) * z / 2 - y) : 0;
  cout << ans;
  return 0;
}
